"""Visual understanding using local Qwen VLM models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from videopython.ai._device import log_device_initialization, select_device
from videopython.base.description import DetectedObject

SCENE_VLM_MODEL_IDS: dict[str, str] = {
    "2b": "Qwen/Qwen3-VL-2B-Instruct",
    "4b": "Qwen/Qwen3-VL-4B-Instruct",
}
DEFAULT_SCENE_VLM_MODEL_SIZE: Literal["2b", "4b"] = "4b"

_DEFAULT_PROMPT = (
    "Analyze the scene across all provided frames. Return exactly one JSON object and nothing else. "
    'Schema: {"caption":"string","primary_action":"string","confidence":0.0,'
    '"objects":[{"label":"string","confidence":0.0}],"text":["string"]}. '
    "Rules: valid JSON only (double quotes, no trailing commas), confidence values must be numbers in [0,1], "
    "objects max 6 unique labels, text max 6 unique entries, keep all strings concise, "
    "and stop generation immediately after the final closing brace."
)


@dataclass
class SceneVLMResult:
    """Structured scene understanding result from the VLM."""

    caption: str
    objects: list[DetectedObject] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    primary_action: str | None = None
    confidence: float = 0.0
    raw_response: str | None = None


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _to_confidence(value: Any, *, fallback: float) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, conf))


def _extract_json(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


class _SceneVLMObjectPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    confidence: float = 0.4


class _SceneVLMResponsePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    caption: str
    primary_action: str | None = None
    confidence: float = 0.35
    objects: list[_SceneVLMObjectPayload] = Field(default_factory=list)
    text: list[str] = Field(default_factory=list)


def _parse_payload(raw: str) -> _SceneVLMResponsePayload | None:
    payload = _extract_json(raw)
    if payload is None:
        return None
    try:
        return _SceneVLMResponsePayload.model_validate(payload)
    except ValidationError:
        return None


class SceneVLM:
    """Generates structured scene understanding with local Qwen3-VL."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        model_size: Literal["2b", "4b"] = DEFAULT_SCENE_VLM_MODEL_SIZE,
    ):
        if model_size not in SCENE_VLM_MODEL_IDS:
            supported = ", ".join(sorted(SCENE_VLM_MODEL_IDS))
            raise ValueError(f"model_size must be one of: {supported}")

        self.model_size = model_size
        self.model_name = model_name or SCENE_VLM_MODEL_IDS[model_size]
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._processor: Any = None
        self._model: Any = None

    def _init_local(self) -> None:
        """Initialize local Qwen3-VL model."""
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore[attr-defined]

        requested_device = self.device
        resolved_device = select_device(self.device, mps_allowed=False)

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(self.model_name, dtype="auto")
        self._model.to(resolved_device)
        self._model.eval()
        self.device = resolved_device

        log_device_initialization(
            "SceneVLM",
            requested_device=requested_device,
            resolved_device=resolved_device,
        )

    def _generation_config_for_run(self) -> Any | None:
        base_config = getattr(self._model, "generation_config", None)
        if base_config is None or not hasattr(base_config, "to_dict"):
            return None

        config = base_config.__class__.from_dict(base_config.to_dict())
        if self.temperature > 0:
            config.do_sample = True
            config.temperature = self.temperature
            return config

        config.do_sample = False
        for name, value in (("temperature", 1.0), ("top_p", 1.0), ("top_k", 50)):
            if hasattr(config, name):
                setattr(config, name, value)
        return config

    def analyze_frame(
        self,
        image: np.ndarray | Image.Image,
        prompt: str | None = None,
    ) -> SceneVLMResult:
        """Analyze one frame and return structured scene understanding."""
        frame = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        return self.analyze_scene([frame], prompt=prompt)

    def analyze_scene(
        self,
        images: list[np.ndarray | Image.Image],
        prompt: str | None = None,
    ) -> SceneVLMResult:
        """Analyze a scene with multiple frames in one multi-image inference."""
        if not images:
            raise ValueError("`images` must contain at least one frame")

        pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]
        user_prompt = prompt or _DEFAULT_PROMPT
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": user_prompt})
        messages = [{"role": "user", "content": content}]
        outputs = self._generate_from_message_batch([messages])
        return self._parse_response(outputs[0])

    def _generate_from_message_batch(self, messages_batch: list[list[dict[str, Any]]]) -> list[str]:
        """Run batch generation for one or more multimodal chat messages."""
        import torch

        if self._model is None:
            self._init_local()

        texts = [
            self._processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]

        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }

        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except ImportError:
            image_inputs = [
                [
                    item["image"]
                    for part in message
                    for item in part.get("content", [])
                    if isinstance(item, dict) and item.get("type") == "image" and "image" in item
                ]
                for message in messages_batch
            ]
            if all(len(items) == 1 for items in image_inputs):
                processor_kwargs["images"] = [items[0] for items in image_inputs]
            else:
                processor_kwargs["images"] = image_inputs
        else:
            image_inputs, video_inputs = process_vision_info(messages_batch)
            processor_kwargs["images"] = image_inputs
            if video_inputs is not None:
                processor_kwargs["videos"] = video_inputs

        inputs = self._processor(**processor_kwargs)
        inputs = inputs.to(self.device) if hasattr(inputs, "to") else {k: v.to(self.device) for k, v in inputs.items()}

        generation_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        generation_config = self._generation_config_for_run()
        if generation_config is not None:
            generation_kwargs["generation_config"] = generation_config
        elif self.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": self.temperature})
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **generation_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], output_ids, strict=False)
        ]
        output_texts = self._processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return [text.strip() for text in output_texts]

    def _parse_response(self, raw: str) -> SceneVLMResult:
        payload = _parse_payload(raw)
        if payload is None:
            raw_text = _normalize_text(raw)
            fallback_caption = raw_text if raw_text and not raw_text.startswith("{") else "No scene description"
            return SceneVLMResult(
                caption=fallback_caption,
                objects=[],
                text=[],
                primary_action=None,
                confidence=0.35,
                raw_response=raw,
            )

        caption = _normalize_text(payload.caption) or "No scene description"
        primary_action = _normalize_text(payload.primary_action) or None
        confidence = _to_confidence(payload.confidence, fallback=0.35)

        objects: list[DetectedObject] = []
        for item in payload.objects:
            label = _normalize_text(item.label)
            if not label:
                continue
            objects.append(
                DetectedObject(
                    label=label,
                    confidence=_to_confidence(item.confidence, fallback=0.4),
                    bounding_box=None,
                )
            )

        text: list[str] = []
        seen: set[str] = set()
        for token_value in payload.text:
            token = _normalize_text(token_value)
            key = token.lower()
            if not token or key in seen:
                continue
            seen.add(key)
            text.append(token)

        return SceneVLMResult(
            caption=caption,
            objects=objects,
            text=text,
            primary_action=primary_action,
            confidence=confidence,
            raw_response=raw,
        )
