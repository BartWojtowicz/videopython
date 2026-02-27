"""Local detection utilities for objects, faces, text, and camera motion."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PIL import Image

from videopython.ai._device import select_device
from videopython.base.description import BoundingBox, DetectedFace, DetectedObject, DetectedText


class ObjectDetector:
    """Detects objects in images using local YOLO models."""

    def __init__(
        self,
        model_size: str = "n",
        confidence_threshold: float = 0.25,
        device: str | None = None,
    ):
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model: Any = None

    def _init_yolo(self) -> None:
        """Initialize YOLO model."""
        from ultralytics import YOLO

        self._model = YOLO(f"yolo11{self.model_size}.pt")
        selected_device = select_device(self.device, mps_allowed=False)
        if selected_device != "cpu":
            self._model.to(selected_device)
        self.device = selected_device

    def detect(self, image: np.ndarray | Image.Image) -> list[DetectedObject]:
        """Detect objects in an image."""
        if self._model is None:
            self._init_yolo()

        img_array = np.array(image) if isinstance(image, Image.Image) else image
        results = self._model(img_array, conf=self.confidence_threshold, verbose=False)

        detected_objects: list[DetectedObject] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            img_h, img_w = result.orig_shape
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                label = self._model.names[cls_id]

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


class FaceDetector:
    """Detects faces in images using OpenCV (CPU) or YOLOv8-face (GPU)."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_face_size: int = 30,
        backend: Literal["cpu", "gpu", "auto"] = "cpu",
        device: str | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.backend: Literal["cpu", "gpu", "auto"] = backend
        self.device = device

        self._cascade: Any = None
        self._yolo_model: Any = None
        self._resolved_backend: Literal["cpu", "gpu"] | None = None

    def _get_device(self) -> str:
        """Get the device to use for GPU inference."""
        return select_device(self.device, mps_allowed=True)

    def _resolve_backend(self) -> Literal["cpu", "gpu"]:
        """Resolve 'auto' backend to an actual backend."""
        if self._resolved_backend is not None:
            return self._resolved_backend

        if self.backend == "auto":
            device = self._get_device()
            self._resolved_backend = "gpu" if device in ("cuda", "mps") else "cpu"
        else:
            self._resolved_backend = self.backend

        return self._resolved_backend

    def _init_cascade(self) -> None:
        """Initialize OpenCV Haar cascade for CPU detection."""
        import cv2

        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _init_yolo_face(self) -> None:
        """Initialize YOLO face detection model for GPU detection."""
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt",
        )
        self._yolo_model = YOLO(model_path)

        device = self._get_device()
        if device != "cpu":
            self._yolo_model.to(device)

    def _detect_cpu(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect faces using OpenCV Haar cascade (CPU)."""
        import cv2

        img_h, img_w = image.shape[:2]

        if len(image.shape) == 3 and image.shape[2] == 3:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image

        if self._cascade is None:
            self._init_cascade()

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )

        detected_faces: list[DetectedFace] = []
        for x, y, w, h in faces:
            bbox = BoundingBox(
                x=x / img_w,
                y=y / img_h,
                width=w / img_w,
                height=h / img_h,
            )
            detected_faces.append(DetectedFace(bounding_box=bbox, confidence=1.0))

        detected_faces.sort(key=lambda f: f.area or 0, reverse=True)
        return detected_faces

    def _detect_gpu(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect faces using YOLOv8-face model (GPU)."""
        if self._yolo_model is None:
            self._init_yolo_face()

        img_h, img_w = image.shape[:2]
        results = self._yolo_model(image, conf=self.confidence_threshold, verbose=False)

        detected_faces: list[DetectedFace] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])

                face_w = x2 - x1
                face_h = y2 - y1
                if face_w < self.min_face_size or face_h < self.min_face_size:
                    continue

                bbox = BoundingBox(
                    x=x1 / img_w,
                    y=y1 / img_h,
                    width=face_w / img_w,
                    height=face_h / img_h,
                )
                detected_faces.append(DetectedFace(bounding_box=bbox, confidence=conf))

        detected_faces.sort(key=lambda f: f.area or 0, reverse=True)
        return detected_faces

    def detect(self, image: np.ndarray | Image.Image) -> list[DetectedFace]:
        """Detect faces in an image."""
        img_array = np.array(image) if isinstance(image, Image.Image) else image

        backend = self._resolve_backend()
        if backend == "gpu":
            return self._detect_gpu(img_array)
        return self._detect_cpu(img_array)

    def detect_batch(self, images: list[np.ndarray] | np.ndarray) -> list[list[DetectedFace]]:
        """Detect faces in a batch of images."""
        backend = self._resolve_backend()

        if isinstance(images, np.ndarray):
            if images.ndim == 4:
                images = [images[i] for i in range(images.shape[0])]
            else:
                images = [images]

        if not images:
            return []

        if backend == "cpu":
            return [self._detect_cpu(img) for img in images]

        if self._yolo_model is None:
            self._init_yolo_face()

        results = self._yolo_model(images, conf=self.confidence_threshold, verbose=False)

        batch_results: list[list[DetectedFace]] = []
        for result in results:
            detected_faces: list[DetectedFace] = []
            boxes = result.boxes
            if boxes is not None:
                result_h, result_w = result.orig_shape

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    conf = float(boxes.conf[i])

                    face_w = x2 - x1
                    face_h = y2 - y1
                    if face_w < self.min_face_size or face_h < self.min_face_size:
                        continue

                    bbox = BoundingBox(
                        x=x1 / result_w,
                        y=y1 / result_h,
                        width=face_w / result_w,
                        height=face_h / result_h,
                    )
                    detected_faces.append(DetectedFace(bounding_box=bbox, confidence=conf))

            detected_faces.sort(key=lambda f: f.area or 0, reverse=True)
            batch_results.append(detected_faces)

        return batch_results


class TextDetector:
    """Detects text in images using local EasyOCR."""

    def __init__(self, languages: list[str] | None = None, device: str | None = None):
        self.languages = languages or ["en"]
        self.device = device
        self._reader: Any = None

    def _init_easyocr(self) -> None:
        """Initialize EasyOCR reader."""
        import easyocr

        selected_device = select_device(self.device, mps_allowed=False)
        self._reader = easyocr.Reader(self.languages, gpu=(selected_device == "cuda"))
        self.device = selected_device

    def detect(self, image: np.ndarray | Image.Image) -> list[str]:
        """Detect text in an image.

        Returns plain text strings for backward compatibility.
        """
        return [item.text for item in self.detect_detailed(image)]

    def detect_detailed(self, image: np.ndarray | Image.Image) -> list[DetectedText]:
        """Detect text in an image with confidence and region boxes."""
        if self._reader is None:
            self._init_easyocr()

        img_array = np.array(image) if isinstance(image, Image.Image) else image
        results = self._reader.readtext(img_array)

        img_h, img_w = img_array.shape[:2]
        detected_text: list[DetectedText] = []
        for polygon, text, confidence in results:
            text_value = str(text).strip()
            if not text_value:
                continue

            bbox: BoundingBox | None = None
            try:
                if polygon:
                    xs = [float(point[0]) for point in polygon]
                    ys = [float(point[1]) for point in polygon]
                    x_min = max(0.0, min(xs))
                    x_max = min(float(img_w), max(xs))
                    y_min = max(0.0, min(ys))
                    y_max = min(float(img_h), max(ys))
                    width = max(0.0, x_max - x_min)
                    height = max(0.0, y_max - y_min)
                    if img_w > 0 and img_h > 0:
                        bbox = BoundingBox(
                            x=x_min / img_w,
                            y=y_min / img_h,
                            width=width / img_w,
                            height=height / img_h,
                        )
            except Exception:
                bbox = None

            detected_text.append(
                DetectedText(
                    text=text_value,
                    confidence=float(confidence),
                    bounding_box=bbox,
                )
            )

        return detected_text


class CameraMotionDetector:
    """Detects camera motion between frames using optical flow."""

    MOTION_TYPES: list[str] = ["static", "pan", "tilt", "zoom", "complex"]

    def __init__(
        self,
        motion_threshold: float = 2.0,
        zoom_threshold: float = 0.1,
    ):
        self.motion_threshold = motion_threshold
        self.zoom_threshold = zoom_threshold

    def detect(
        self,
        frame1: np.ndarray | Image.Image,
        frame2: np.ndarray | Image.Image,
    ) -> str:
        """Detect camera motion between two consecutive frames."""
        import cv2

        img1 = np.array(frame1) if isinstance(frame1, Image.Image) else frame1
        img2 = np.array(frame2) if isinstance(frame2, Image.Image) else frame2

        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2

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

        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        avg_magnitude = np.mean(magnitude)

        if avg_magnitude < self.motion_threshold:
            return "static"

        mean_flow_x = np.mean(flow_x)
        mean_flow_y = np.mean(flow_y)

        h, w = gray1.shape
        cy, cx = h // 2, w // 2

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

        if edge_mag > 0 and abs(edge_mag - center_mag) / edge_mag > self.zoom_threshold:
            return "zoom"

        abs_x = abs(mean_flow_x)
        abs_y = abs(mean_flow_y)

        if abs_x > abs_y * 1.5:
            return "pan"
        elif abs_y > abs_x * 1.5:
            return "tilt"
        else:
            return "complex"
