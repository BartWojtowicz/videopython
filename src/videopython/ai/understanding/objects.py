"""General object detection for the understanding layer.

``ObjectDetector`` is the object-detection counterpart to the face detector in
``faces.py``: a lazy D-FINE COCO detector (transformers) returning
:class:`~videopython.base.description.DetectedObject` with normalized bounding
boxes. Both share :class:`~videopython.ai.understanding._detector.DetectorBase`
(lazy init, device selection, ``detect`` / ``detect_batch`` / ``unload``), so the
two stay one mental model. Consumed by
``videopython.ai.effects.ObjectDetectionOverlay``; usable directly for any
per-frame object analysis.

D-FINE (Apache-2.0) replaced the AGPL-licensed Ultralytics YOLO weights. Its COCO
labels use VOC-style spellings for six classes (``motorbike``, ``aeroplane``,
``sofa``, ``pottedplant``, ``diningtable``, ``tvmonitor``) where the standard COCO
names are ``motorcycle``, ``airplane``, ``couch``, ``potted plant``,
``dining table`` and ``tv``. ``class_filter`` accepts either spelling: names are
normalized through :data:`CLASS_ALIASES` before matching, and any name the model
does not emit is logged once the class list is known.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

from videopython.ai._revisions import pinned
from videopython.ai.understanding._detector import Backend, DetectorBase
from videopython.base.description import BoundingBox, DetectedObject

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ObjectDetector", "MODEL_SIZES", "CLASS_ALIASES", "normalize_class_names"]

# D-FINE emits VOC-style spellings for six COCO classes. Callers -- and LLMs asked
# to name a class -- reach for the standard COCO spelling, which would match
# nothing and draw nothing, with no error to explain the silence. Accept both.
CLASS_ALIASES: dict[str, str] = {
    "motorcycle": "motorbike",
    "airplane": "aeroplane",
    "couch": "sofa",
    "potted plant": "pottedplant",
    "dining table": "diningtable",
    "tv": "tvmonitor",
}


def normalize_class_names(names: "Iterable[str]") -> tuple[str, ...]:
    """Map COCO-standard class names onto the VOC-style spellings D-FINE emits.

    Case and surrounding whitespace are normalized first, so ``"Potted Plant"``
    and ``"potted  plant"`` both reach ``pottedplant``. A name with no alias is
    passed through unchanged (lowercased) -- this translates spellings, it does
    not validate them; see :meth:`ObjectDetector.unknown_filter_classes`.
    """
    return tuple(CLASS_ALIASES.get(key, key) for key in (" ".join(n.lower().split()) for n in names))


# D-FINE COCO checkpoints (Apache-2.0), in ascending size/quality. ``model_size``
# on ``ObjectDetectionOverlay`` maps onto these; pinned in ``_revisions.py``.
MODEL_SIZES: dict[str, str] = {
    "n": "ustc-community/dfine-nano-coco",
    "s": "ustc-community/dfine-small-coco",
    "m": "ustc-community/dfine-medium-coco",
}
DEFAULT_MODEL = MODEL_SIZES["n"]


class ObjectDetector(DetectorBase[DetectedObject]):
    """Lazy D-FINE COCO object detector returning normalized detections.

    The D-FINE weights (default ``ustc-community/dfine-nano-coco``) download from
    HuggingFace on first real use; class names come from the model config.
    Detection is gated by ``confidence_threshold`` and optionally restricted to
    ``class_filter``, which accepts either D-FINE's VOC-style spellings or the
    standard COCO ones (``motorcycle``, ``airplane``, ``couch``, ``potted plant``,
    ``dining table``, ``tv``) -- see :func:`normalize_class_names`.
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    _FEATURE = "ObjectDetector"
    # Override the base sentinel/unload set: hold the model AND the processor.
    _model_attrs = ("_model", "_processor")

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        class_filter: tuple[str, ...] = (),
        backend: Backend = "auto",
    ):
        """Initialize the detector.

        Args:
            model_name: D-FINE COCO HuggingFace repo id (e.g.
                ``ustc-community/dfine-nano-coco``, ``...-small-coco``,
                ``...-medium-coco``, ``...-large-coco``). Downloaded on first use.
            confidence_threshold: Minimum detection confidence in ``[0, 1]``.
            class_filter: If non-empty, only these COCO class names are kept.
                Either spelling works -- ``motorcycle`` and ``motorbike`` both
                match -- and names are normalized via
                :func:`normalize_class_names`. A name the model never emits is
                logged as a warning once the model loads, since it would
                otherwise just silently match nothing.
            backend: Detection device - ``"cpu"``, ``"gpu"``, or ``"auto"``.
        """
        super().__init__(backend=backend)
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.class_filter = normalize_class_names(class_filter)
        self._model: Any = None
        self._processor: Any = None
        self._class_names: dict[int, str] = {}
        logger.info("ObjectDetector initialized with model=%s backend=%s", model_name, backend)

    def _load_model(self) -> None:
        from videopython.ai._optional import require

        tf = require("transformers", feature=self._FEATURE)
        revision = pinned(self.model_name)
        self._processor = tf.AutoImageProcessor.from_pretrained(self.model_name, revision=revision, use_fast=True)
        model = tf.DFineForObjectDetection.from_pretrained(self.model_name, revision=revision)
        model.eval()
        if self._resolve_device() == "cuda":
            model = model.to("cuda")
        self._model = model
        self._class_names = {int(k): v for k, v in model.config.id2label.items()}
        self._warn_unknown_filter_classes()

    def unknown_filter_classes(self) -> tuple[str, ...]:
        """``class_filter`` names this model never emits (empty until it loads)."""
        if not self._class_names:
            return ()
        known = set(self._class_names.values())
        return tuple(name for name in self.class_filter if name not in known)

    def _warn_unknown_filter_classes(self) -> None:
        """Log filter names that cannot match, which would otherwise draw nothing."""
        unknown = self.unknown_filter_classes()
        if unknown:
            logger.warning(
                "class_filter names not emitted by %s: %s. Nothing will be detected for them; "
                "the model's classes are %s.",
                self.model_name,
                ", ".join(unknown),
                ", ".join(sorted(set(self._class_names.values()))),
            )

    def _infer(self, images: list[np.ndarray]) -> list[list[DetectedObject]]:
        import torch

        device = self._resolve_device()
        inputs = self._processor(images=images, return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = self._model(**inputs)
        # target_sizes is (height, width) per image; D-FINE letterboxes internally
        # so post-processing needs the original sizes to de-letterbox the boxes.
        target_sizes = torch.tensor([[img.shape[0], img.shape[1]] for img in images], device=device)
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )
        return [self._parse(result, img.shape[1], img.shape[0]) for result, img in zip(results, images)]

    def _parse(self, result: dict[str, Any], img_w: int, img_h: int) -> list[DetectedObject]:
        detected: list[DetectedObject] = []
        scores = result["scores"].tolist()
        labels = result["labels"].tolist()
        boxes = result["boxes"].tolist()
        for score, label_id, (x1, y1, x2, y2) in zip(scores, labels, boxes):
            label = self._class_names.get(int(label_id), str(int(label_id)))
            if self.class_filter and label not in self.class_filter:
                continue
            # D-FINE boxes can sit slightly outside the frame; clamp before normalizing.
            x1 = min(max(x1, 0.0), img_w)
            x2 = min(max(x2, 0.0), img_w)
            y1 = min(max(y1, 0.0), img_h)
            y2 = min(max(y2, 0.0), img_h)
            detected.append(
                DetectedObject(
                    label=label,
                    confidence=float(score),
                    bounding_box=BoundingBox(
                        x=x1 / img_w,
                        y=y1 / img_h,
                        width=(x2 - x1) / img_w,
                        height=(y2 - y1) / img_h,
                    ),
                )
            )
        detected.sort(key=lambda d: d.confidence, reverse=True)
        return detected
