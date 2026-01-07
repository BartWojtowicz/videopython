# Transitions

Transitions combine two videos with visual effects at the boundary.

## Usage

```python
from videopython.base import Video, FadeTransition

video1 = Video.from_path("clip1.mp4")
video2 = Video.from_path("clip2.mp4")

fade = FadeTransition(effect_time_seconds=1.5)
combined = fade.apply((video1, video2))
```

!!! note "Requirements"
    Both videos must have the same dimensions and frame rate to be combined.

## Transition (Base Class)

::: videopython.base.Transition

## FadeTransition

::: videopython.base.FadeTransition

## BlurTransition

::: videopython.base.BlurTransition

## InstantTransition

::: videopython.base.InstantTransition
