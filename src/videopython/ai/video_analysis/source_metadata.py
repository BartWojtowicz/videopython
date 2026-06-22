"""Container-metadata parsing for VideoAnalyzer source records.

ffmpeg tag probing, creation-time normalization, and ISO-6709 geo parsing —
plus a small :func:`try_init` helper for best-effort predictor construction.
Lifted out of ``analyzer.py`` so that file stays focused on orchestration.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

from videopython.ai.video_analysis.models import GeoMetadata
from videopython.base import _ffmpeg
from videopython.base.exceptions import FFmpegProbeError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_CREATION_TIME_TAG_KEYS: tuple[str, ...] = (
    "creation_time",
    "com.apple.quicktime.creationdate",
    "date",
)

_GEO_TAG_KEYS: tuple[str, ...] = (
    "com.apple.quicktime.location.iso6709",
    "location",
    "location-eng",
)


def try_init(factory: Callable[[], T], name: str) -> T | None:
    """Construct a component, or log and return ``None`` on a load/runtime error.

    Best-effort init for the optional per-scene predictors: a missing extra or a
    model-load failure degrades that analyzer to "skipped" instead of aborting
    the whole analysis run.
    """
    try:
        return factory()
    except (ImportError, OSError, RuntimeError, ValueError, TypeError):
        logger.warning("Failed to initialize %s, skipping", name, exc_info=True)
        return None


def extract_source_tags(path: Path | None) -> dict[str, str]:
    """Lowercased format + stream tags from ``path`` via ffprobe; ``{}`` on failure."""
    if path is None:
        return {}

    try:
        payload = _ffmpeg.probe(path, extra_args=["-show_entries", "format_tags:stream_tags"])
    except (FFmpegProbeError, OSError):
        return {}

    tags: dict[str, str] = {}

    format_tags = payload.get("format", {}).get("tags", {})
    if isinstance(format_tags, dict):
        tags.update({str(k).lower(): str(v) for k, v in format_tags.items()})

    for stream in payload.get("streams", []):
        stream_tags = stream.get("tags", {})
        if not isinstance(stream_tags, dict):
            continue
        for key, value in stream_tags.items():
            lowered = str(key).lower()
            tags.setdefault(lowered, str(value))

    return tags


def creation_time_from_tags(tags: dict[str, str]) -> str | None:
    """Normalized ISO creation time from the first matching tag key, or ``None``."""
    return _normalize_creation_time(next((tags[key] for key in _CREATION_TIME_TAG_KEYS if key in tags), None))


def _normalize_creation_time(value: str | None) -> str | None:
    if value is None:
        return None

    raw = value.strip()
    if not raw:
        return None

    candidate = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return raw

    if parsed.tzinfo is None:
        return parsed.isoformat()

    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_geo_metadata(tags: dict[str, str]) -> GeoMetadata | None:
    for key in _GEO_TAG_KEYS:
        value = tags.get(key)
        if not value:
            continue
        geo = _parse_iso6709_or_pair(value)
        if geo is not None:
            geo.source = key
            return geo
    return None


def _parse_iso6709_or_pair(value: str) -> GeoMetadata | None:
    iso6709_match = re.match(
        r"^\s*([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?/?\s*$",
        value,
    )
    if iso6709_match:
        lat = float(iso6709_match.group(1))
        lon = float(iso6709_match.group(2))
        alt = float(iso6709_match.group(3)) if iso6709_match.group(3) is not None else None
        return GeoMetadata(latitude=lat, longitude=lon, altitude=alt)

    pair_match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(?:\s*,\s*(-?\d+(?:\.\d+)?))?\s*$", value)
    if pair_match:
        lat = float(pair_match.group(1))
        lon = float(pair_match.group(2))
        alt = float(pair_match.group(3)) if pair_match.group(3) is not None else None
        return GeoMetadata(latitude=lat, longitude=lon, altitude=alt)

    return None
