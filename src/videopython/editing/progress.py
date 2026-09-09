from dataclasses import dataclass
from time import monotonic
from typing import Callable, Literal

RenderStage = Literal["compilation", "segment", "assembly", "post_operations", "audio_mix", "complete"]


@dataclass(frozen=True)
class RenderProgress:
    """Work completed within one render stage; only ``complete`` means job success."""

    stage: RenderStage
    completed: int
    total: int | None
    unit: Literal["frames", "steps"]
    segment_index: int | None
    finished: bool


class _Progress:
    def __init__(self, callback: Callable[[RenderProgress], None] | None):
        self.callback = callback
        self.last_update = 0.0

    def start(
        self,
        stage: RenderStage,
        *,
        total: int | None = 1,
        unit: Literal["frames", "steps"] = "steps",
        segment_index: int | None = None,
    ) -> None:
        self.stage = stage
        self.total = total
        self.unit = unit
        self.segment_index = segment_index
        self.completed = 0
        self._emit(False)

    def advance(self, completed: int) -> None:
        self.completed = completed
        if monotonic() - self.last_update >= 0.25:
            self._emit(False)

    def finish(self) -> None:
        if self.unit == "steps":
            self.completed = 1
        self._emit(True)

    def _emit(self, finished: bool) -> None:
        if self.callback is not None:
            self.callback(
                RenderProgress(self.stage, self.completed, self.total, self.unit, self.segment_index, finished)
            )
        self.last_update = monotonic()
