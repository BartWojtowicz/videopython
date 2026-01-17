"""Detection backends for object detection, face detection, OCR, and frame analysis."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from videopython.ai.backends import ImageToTextBackend, UnsupportedBackendError, get_api_key
from videopython.ai.config import get_default_backend
from videopython.base.description import BoundingBox, DetectedFace, DetectedObject


class ObjectDetector:
    """Detects objects in images using YOLO (local) or vision LLMs (cloud)."""

    SUPPORTED_BACKENDS: list[str] = ["local", "openai", "gemini"]

    def __init__(
        self,
        backend: ImageToTextBackend | None = None,
        model_size: str = "n",
        confidence_threshold: float = 0.25,
        api_key: str | None = None,
    ):
        """Initialize object detector.

        Args:
            backend: Backend to use ('local' for YOLO, 'openai'/'gemini' for vision LLMs).
            model_size: YOLO model size for local backend ('n', 's', 'm', 'l', 'x').
            confidence_threshold: Minimum confidence for detections (0-1).
            api_key: API key for cloud backends.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("image_to_text")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: ImageToTextBackend = resolved_backend  # type: ignore[assignment]
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.api_key = api_key
        self._model: Any = None

    def _init_yolo(self) -> None:
        """Initialize YOLO model."""
        from ultralytics import YOLO

        self._model = YOLO(f"yolo11{self.model_size}.pt")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _detect_local(self, image: np.ndarray | Image.Image) -> list[DetectedObject]:
        """Detect objects using YOLO."""
        if self._model is None:
            self._init_yolo()

        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        results = self._model(img_array, conf=self.confidence_threshold, verbose=False)
        detected_objects = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            img_h, img_w = result.orig_shape

            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                label = self._model.names[cls_id]

                # Normalize coordinates to [0, 1]
                bbox = BoundingBox(
                    x=x1 / img_w,
                    y=y1 / img_h,
                    width=(x2 - x1) / img_w,
                    height=(y2 - y1) / img_h,
                )

                detected_objects.append(
                    DetectedObject(
                        label=label,
                        confidence=conf,
                        bounding_box=bbox,
                    )
                )

        return detected_objects

    def _detect_openai(self, image: np.ndarray | Image.Image) -> list[DetectedObject]:
        """Detect objects using OpenAI GPT-4o."""
        from openai import OpenAI

        api_key = get_api_key("openai", self.api_key)
        client = OpenAI(api_key=api_key)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_base64 = self._image_to_base64(image)

        prompt = """Analyze this image and detect all visible objects.
Return a JSON array of objects with this exact format:
[{"label": "object name", "confidence": 0.95, "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}}]

Where bbox coordinates are normalized (0-1) relative to image dimensions:
- x: left edge of bounding box
- y: top edge of bounding box
- width: width of box
- height: height of box

Only include objects you're confident about. Return empty array [] if no objects detected.
Return ONLY the JSON array, no other text."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ],
            max_tokens=1000,
        )

        return self._parse_detection_response(response.choices[0].message.content or "[]")

    def _detect_gemini(self, image: np.ndarray | Image.Image) -> list[DetectedObject]:
        """Detect objects using Google Gemini."""
        import google.generativeai as genai

        api_key = get_api_key("gemini", self.api_key)
        genai.configure(api_key=api_key)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = """Analyze this image and detect all visible objects.
Return a JSON array of objects with this exact format:
[{"label": "object name", "confidence": 0.95, "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}}]

Where bbox coordinates are normalized (0-1) relative to image dimensions:
- x: left edge of bounding box
- y: top edge of bounding box
- width: width of box
- height: height of box

Only include objects you're confident about. Return empty array [] if no objects detected.
Return ONLY the JSON array, no other text."""

        response = model.generate_content([prompt, image])
        return self._parse_detection_response(response.text)

    def _parse_detection_response(self, response: str) -> list[DetectedObject]:
        """Parse JSON response from cloud backends into DetectedObject list."""
        try:
            # Clean up response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            response = response.strip()

            data = json.loads(response)
            detected_objects = []

            for obj in data:
                bbox = None
                if "bbox" in obj and obj["bbox"]:
                    bbox = BoundingBox(
                        x=float(obj["bbox"]["x"]),
                        y=float(obj["bbox"]["y"]),
                        width=float(obj["bbox"]["width"]),
                        height=float(obj["bbox"]["height"]),
                    )

                detected_objects.append(
                    DetectedObject(
                        label=obj["label"],
                        confidence=float(obj.get("confidence", 0.8)),
                        bounding_box=bbox,
                    )
                )

            return detected_objects
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def detect(self, image: np.ndarray | Image.Image) -> list[DetectedObject]:
        """Detect objects in an image.

        Args:
            image: Image as numpy array (H, W, 3) in RGB format or PIL Image.

        Returns:
            List of DetectedObject instances.
        """
        if self.backend == "local":
            return self._detect_local(image)
        elif self.backend == "openai":
            return self._detect_openai(image)
        elif self.backend == "gemini":
            return self._detect_gemini(image)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)


class FaceDetector:
    """Detects faces in images using OpenCV Haar cascade."""

    def __init__(self, confidence_threshold: float = 0.5, min_face_size: int = 30):
        """Initialize face detector.

        Args:
            confidence_threshold: Minimum confidence for detections (0-1).
            min_face_size: Minimum face size in pixels.
        """
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self._cascade: Any = None
        self._model_loaded = False

    def _init_cascade(self) -> None:
        """Initialize OpenCV Haar cascade."""
        import cv2

        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._model_loaded = True

    def detect(self, image: np.ndarray | Image.Image) -> list[DetectedFace]:
        """Detect faces in an image.

        Args:
            image: Image as numpy array (H, W, 3) in RGB format or PIL Image.

        Returns:
            List of DetectedFace objects with bounding boxes.
        """
        import cv2

        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        img_h, img_w = img_array.shape[:2]

        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        if not self._model_loaded:
            self._init_cascade()

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )

        detected_faces = []
        for x, y, w, h in faces:
            # Normalize coordinates to [0, 1]
            bbox = BoundingBox(
                x=x / img_w,
                y=y / img_h,
                width=w / img_w,
                height=h / img_h,
            )
            detected_faces.append(DetectedFace(bounding_box=bbox, confidence=1.0))

        # Sort by area (largest first) - all faces from local detector have bounding boxes
        detected_faces.sort(key=lambda f: f.area or 0, reverse=True)
        return detected_faces


class TextDetector:
    """Detects text in images using EasyOCR (local) or vision LLMs (cloud)."""

    SUPPORTED_BACKENDS: list[str] = ["local", "openai", "gemini"]

    def __init__(
        self,
        backend: ImageToTextBackend | None = None,
        languages: list[str] | None = None,
        api_key: str | None = None,
    ):
        """Initialize text detector.

        Args:
            backend: Backend to use ('local' for EasyOCR, 'openai'/'gemini' for vision LLMs).
            languages: List of language codes for EasyOCR (default: ['en']).
            api_key: API key for cloud backends.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("image_to_text")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: ImageToTextBackend = resolved_backend  # type: ignore[assignment]
        self.languages = languages or ["en"]
        self.api_key = api_key
        self._reader: Any = None

    def _init_easyocr(self) -> None:
        """Initialize EasyOCR reader."""
        import easyocr

        self._reader = easyocr.Reader(self.languages, gpu=False)

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _detect_local(self, image: np.ndarray | Image.Image) -> list[str]:
        """Detect text using EasyOCR."""
        if self._reader is None:
            self._init_easyocr()

        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        results = self._reader.readtext(img_array)
        # Extract just the text from results (each result is [bbox, text, confidence])
        return [text for _, text, _ in results if text.strip()]

    def _detect_openai(self, image: np.ndarray | Image.Image) -> list[str]:
        """Detect text using OpenAI GPT-4o."""
        from openai import OpenAI

        api_key = get_api_key("openai", self.api_key)
        client = OpenAI(api_key=api_key)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_base64 = self._image_to_base64(image)

        prompt = """Extract all visible text from this image.
Return a JSON array of strings, where each string is a distinct piece of text found in the image.
Example: ["STOP", "Main Street", "Open 24 Hours"]
Return empty array [] if no text is found.
Return ONLY the JSON array, no other text."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ],
            max_tokens=500,
        )

        return self._parse_text_response(response.choices[0].message.content or "[]")

    def _detect_gemini(self, image: np.ndarray | Image.Image) -> list[str]:
        """Detect text using Google Gemini."""
        import google.generativeai as genai

        api_key = get_api_key("gemini", self.api_key)
        genai.configure(api_key=api_key)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = """Extract all visible text from this image.
Return a JSON array of strings, where each string is a distinct piece of text found in the image.
Example: ["STOP", "Main Street", "Open 24 Hours"]
Return empty array [] if no text is found.
Return ONLY the JSON array, no other text."""

        response = model.generate_content([prompt, image])
        return self._parse_text_response(response.text)

    def _parse_text_response(self, response: str) -> list[str]:
        """Parse JSON response from cloud backends into string list."""
        try:
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            response = response.strip()

            data = json.loads(response)
            return [str(text) for text in data if text]
        except (json.JSONDecodeError, TypeError):
            return []

    def detect(self, image: np.ndarray | Image.Image) -> list[str]:
        """Detect text in an image.

        Args:
            image: Image as numpy array (H, W, 3) in RGB format or PIL Image.

        Returns:
            List of detected text strings.
        """
        if self.backend == "local":
            return self._detect_local(image)
        elif self.backend == "openai":
            return self._detect_openai(image)
        elif self.backend == "gemini":
            return self._detect_gemini(image)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)


class ShotTypeClassifier:
    """Classifies shot types using vision LLMs."""

    SUPPORTED_BACKENDS: list[str] = ["openai", "gemini"]
    SHOT_TYPES: list[str] = ["extreme-wide", "wide", "medium", "medium-close-up", "close-up", "extreme-close-up"]

    def __init__(
        self,
        backend: ImageToTextBackend | None = None,
        api_key: str | None = None,
    ):
        """Initialize shot type classifier.

        Args:
            backend: Backend to use ('openai' or 'gemini').
            api_key: API key for cloud backends.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("image_to_text")
        # Default to openai if local is configured (no local backend for shot type)
        if resolved_backend == "local":
            resolved_backend = "openai"
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: ImageToTextBackend = resolved_backend  # type: ignore[assignment]
        self.api_key = api_key

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _classify_openai(self, image: np.ndarray | Image.Image) -> str | None:
        """Classify shot type using OpenAI GPT-4o."""
        from openai import OpenAI

        api_key = get_api_key("openai", self.api_key)
        client = OpenAI(api_key=api_key)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_base64 = self._image_to_base64(image)

        prompt = f"""Classify the cinematographic shot type of this image.
Choose exactly one from: {", ".join(self.SHOT_TYPES)}

Definitions:
- extreme-wide: Very distant view, landscape or establishing shot
- wide: Full scene visible, subjects appear small
- medium: Subject from waist/knees up
- medium-close-up: Subject from chest up
- close-up: Face or object fills most of frame
- extreme-close-up: Detail shot, part of face or small object

Return ONLY the shot type label, nothing else."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ],
            max_tokens=50,
        )

        result = (response.choices[0].message.content or "").strip().lower()
        return result if result in self.SHOT_TYPES else None

    def _classify_gemini(self, image: np.ndarray | Image.Image) -> str | None:
        """Classify shot type using Google Gemini."""
        import google.generativeai as genai

        api_key = get_api_key("gemini", self.api_key)
        genai.configure(api_key=api_key)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""Classify the cinematographic shot type of this image.
Choose exactly one from: {", ".join(self.SHOT_TYPES)}

Definitions:
- extreme-wide: Very distant view, landscape or establishing shot
- wide: Full scene visible, subjects appear small
- medium: Subject from waist/knees up
- medium-close-up: Subject from chest up
- close-up: Face or object fills most of frame
- extreme-close-up: Detail shot, part of face or small object

Return ONLY the shot type label, nothing else."""

        response = model.generate_content([prompt, image])
        result = response.text.strip().lower()
        return result if result in self.SHOT_TYPES else None

    def classify(self, image: np.ndarray | Image.Image) -> str | None:
        """Classify the shot type of an image.

        Args:
            image: Image as numpy array (H, W, 3) in RGB format or PIL Image.

        Returns:
            Shot type string or None if classification failed.
        """
        if self.backend == "openai":
            return self._classify_openai(image)
        elif self.backend == "gemini":
            return self._classify_gemini(image)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)


@dataclass
class CombinedFrameAnalysis:
    """Results from combined frame analysis using a single API call."""

    detected_objects: list[DetectedObject]
    detected_text: list[str]
    face_count: int
    shot_type: str | None


class CombinedFrameAnalyzer:
    """Analyzes frames using a single vision API call for efficiency.

    For cloud backends (OpenAI/Gemini), combines object detection, OCR, face counting,
    and shot type classification into a single API call instead of multiple calls.

    Uses structured outputs (JSON schema) to ensure valid responses.
    """

    SUPPORTED_BACKENDS: list[str] = ["openai", "gemini"]
    SHOT_TYPES: list[str] = ["extreme-wide", "wide", "medium", "medium-close-up", "close-up", "extreme-close-up"]

    # JSON Schema for structured output
    RESPONSE_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Object name/class"},
                        "confidence": {"type": "number", "description": "Detection confidence 0-1"},
                        "bbox": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number", "description": "Left edge, normalized 0-1"},
                                "y": {"type": "number", "description": "Top edge, normalized 0-1"},
                                "width": {"type": "number", "description": "Width, normalized 0-1"},
                                "height": {"type": "number", "description": "Height, normalized 0-1"},
                            },
                            "required": ["x", "y", "width", "height"],
                        },
                    },
                    "required": ["label", "confidence"],
                },
            },
            "text": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Text strings found in the image via OCR",
            },
            "face_count": {"type": "integer", "description": "Number of human faces detected"},
            "shot_type": {
                "type": "string",
                "enum": ["extreme-wide", "wide", "medium", "medium-close-up", "close-up", "extreme-close-up"],
                "description": "Cinematographic shot type classification",
            },
        },
        "required": ["objects", "text", "face_count", "shot_type"],
    }

    def __init__(
        self,
        backend: ImageToTextBackend | None = None,
        api_key: str | None = None,
    ):
        """Initialize combined frame analyzer.

        Args:
            backend: Backend to use ('openai' or 'gemini').
            api_key: API key for cloud backends.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("image_to_text")
        if resolved_backend == "local":
            raise UnsupportedBackendError(
                "local", self.SUPPORTED_BACKENDS + [" (use individual detectors for local backend)"]
            )
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: ImageToTextBackend = resolved_backend  # type: ignore[assignment]
        self.api_key = api_key

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _get_combined_prompt(self) -> str:
        """Get the prompt for combined analysis."""
        return """Analyze this image and extract:
1. All visible objects with their bounding boxes (normalized 0-1 coordinates)
2. Any text visible in the image (OCR)
3. Count of human faces
4. Cinematographic shot type classification

Shot type definitions:
- extreme-wide: Very distant view, landscape or establishing shot
- wide: Full scene visible, subjects appear small
- medium: Subject from waist/knees up
- medium-close-up: Subject from chest up
- close-up: Face or object fills most of frame
- extreme-close-up: Detail shot, part of face or small object"""

    def _parse_response(self, data: dict[str, Any]) -> CombinedFrameAnalysis:
        """Parse structured response into CombinedFrameAnalysis."""
        try:
            # Parse objects
            detected_objects = []
            for obj in data.get("objects", []):
                bbox = None
                if "bbox" in obj and obj["bbox"]:
                    bbox = BoundingBox(
                        x=float(obj["bbox"].get("x", 0)),
                        y=float(obj["bbox"].get("y", 0)),
                        width=float(obj["bbox"].get("width", 0)),
                        height=float(obj["bbox"].get("height", 0)),
                    )
                detected_objects.append(
                    DetectedObject(
                        label=obj.get("label", "unknown"),
                        confidence=float(obj.get("confidence", 0.8)),
                        bounding_box=bbox,
                    )
                )

            # Parse text
            detected_text = [str(t) for t in data.get("text", []) if t]

            # Parse face count
            face_count = int(data.get("face_count", 0))

            # Parse shot type
            shot_type = data.get("shot_type", "").lower() if data.get("shot_type") else None
            if shot_type and shot_type not in self.SHOT_TYPES:
                shot_type = None

            return CombinedFrameAnalysis(
                detected_objects=detected_objects,
                detected_text=detected_text,
                face_count=face_count,
                shot_type=shot_type,
            )
        except (KeyError, TypeError, ValueError):
            return CombinedFrameAnalysis(
                detected_objects=[],
                detected_text=[],
                face_count=0,
                shot_type=None,
            )

    def _analyze_openai(self, image: np.ndarray | Image.Image) -> CombinedFrameAnalysis:
        """Analyze image using OpenAI GPT-4o with structured outputs."""
        from openai import OpenAI

        api_key = get_api_key("openai", self.api_key)
        client = OpenAI(api_key=api_key)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_base64 = self._image_to_base64(image)
        prompt = self._get_combined_prompt()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "frame_analysis",
                    "strict": True,
                    "schema": self.RESPONSE_SCHEMA,
                },
            },
            max_tokens=1000,
        )

        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return self._parse_response(data)

    def _analyze_gemini(self, image: np.ndarray | Image.Image) -> CombinedFrameAnalysis:
        """Analyze image using Google Gemini with structured outputs."""
        import google.generativeai as genai

        api_key = get_api_key("gemini", self.api_key)
        genai.configure(api_key=api_key)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        prompt = self._get_combined_prompt()

        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=self.RESPONSE_SCHEMA,
            ),
        )
        response = model.generate_content([prompt, image])
        data = json.loads(response.text)
        return self._parse_response(data)

    def analyze(self, image: np.ndarray | Image.Image) -> CombinedFrameAnalysis:
        """Analyze an image with a single API call.

        Args:
            image: Image as numpy array (H, W, 3) in RGB format or PIL Image.

        Returns:
            CombinedFrameAnalysis with all detection results.
        """
        if self.backend == "openai":
            return self._analyze_openai(image)
        elif self.backend == "gemini":
            return self._analyze_gemini(image)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)


class CameraMotionDetector:
    """Detects camera motion between frames using optical flow."""

    MOTION_TYPES: list[str] = ["static", "pan", "tilt", "zoom", "complex"]

    def __init__(
        self,
        motion_threshold: float = 2.0,
        zoom_threshold: float = 0.1,
    ):
        """Initialize camera motion detector.

        Args:
            motion_threshold: Minimum average flow magnitude to consider as motion.
            zoom_threshold: Threshold for detecting zoom (relative change in flow magnitude from center).
        """
        self.motion_threshold = motion_threshold
        self.zoom_threshold = zoom_threshold

    def detect(
        self,
        frame1: np.ndarray | Image.Image,
        frame2: np.ndarray | Image.Image,
    ) -> str:
        """Detect camera motion between two consecutive frames.

        Args:
            frame1: First frame as numpy array or PIL Image.
            frame2: Second frame as numpy array or PIL Image.

        Returns:
            Motion type: 'static', 'pan', 'tilt', 'zoom', or 'complex'.
        """
        import cv2

        # Convert to numpy if needed
        if isinstance(frame1, Image.Image):
            img1 = np.array(frame1)
        else:
            img1 = frame1

        if isinstance(frame2, Image.Image):
            img2 = np.array(frame2)
        else:
            img2 = frame2

        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = img1

        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = img2

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Analyze flow vectors
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        # Calculate magnitude
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        avg_magnitude = np.mean(magnitude)

        if avg_magnitude < self.motion_threshold:
            return "static"

        # Calculate mean flow direction
        mean_flow_x = np.mean(flow_x)
        mean_flow_y = np.mean(flow_y)

        # Check for zoom by analyzing flow from center
        h, w = gray1.shape
        cy, cx = h // 2, w // 2

        # Sample flow at different distances from center
        center_region = magnitude[cy - h // 4 : cy + h // 4, cx - w // 4 : cx + w // 4]
        edge_region_top = magnitude[: h // 4, :]
        edge_region_bottom = magnitude[-h // 4 :, :]
        edge_region_left = magnitude[:, : w // 4]
        edge_region_right = magnitude[:, -w // 4 :]

        center_mag = np.mean(center_region) if center_region.size > 0 else 0
        edge_mag = np.mean(
            [
                np.mean(edge_region_top) if edge_region_top.size > 0 else 0,
                np.mean(edge_region_bottom) if edge_region_bottom.size > 0 else 0,
                np.mean(edge_region_left) if edge_region_left.size > 0 else 0,
                np.mean(edge_region_right) if edge_region_right.size > 0 else 0,
            ]
        )

        # Zoom detection: edges move more than center (zoom in) or vice versa
        if edge_mag > 0 and abs(edge_mag - center_mag) / edge_mag > self.zoom_threshold:
            return "zoom"

        # Determine dominant motion direction
        abs_x = abs(mean_flow_x)
        abs_y = abs(mean_flow_y)

        if abs_x > abs_y * 1.5:
            return "pan"  # Horizontal motion
        elif abs_y > abs_x * 1.5:
            return "tilt"  # Vertical motion
        else:
            return "complex"  # Mixed motion
