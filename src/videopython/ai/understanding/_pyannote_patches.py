from __future__ import annotations

import math
from types import MethodType
from typing import Any

import numpy as np


def _shared_embedding_batch(embedding: Any, batch: list[tuple[int, int, Any, Any]], masks: Any) -> np.ndarray:
    """Run a backend's mask-independent frame extractor once per distinct chunk.

    Keep pooling in the original pair order.
    Only the convolutional work is shared; every speaker keeps its own mask.
    """
    import torch

    unique: dict[int, tuple[int, Any]] = {}
    indices = []
    for chunk_index, _, waveform, _ in batch:
        if chunk_index not in unique:
            unique[chunk_index] = (len(unique), waveform)
        indices.append(unique[chunk_index][0])
    with torch.inference_mode():
        frames = embedding.model_.forward_frames(
            torch.vstack([waveform for _, waveform in unique.values()]).to(embedding.device)
        )
        vectors = embedding.model_.forward_embedding(frames[indices], weights=masks.to(embedding.device))
    return vectors.cpu().numpy()


def _reconstruct(pipeline: Any, segmentations: Any, hard_clusters: Any, count: Any) -> Any:
    """Reconstruct global clusters without promoting segmentation data to float64."""
    num_chunks, num_frames, _ = segmentations.data.shape
    num_clusters = int(np.max(hard_clusters)) + 1
    clustered_data = np.full(
        (num_chunks, num_frames, num_clusters),
        np.nan,
        dtype=segmentations.data.dtype,
    )

    for chunk_index, (cluster, (_, segmentation)) in enumerate(zip(hard_clusters, segmentations)):
        for cluster_index in np.unique(cluster):
            if cluster_index == -2:
                continue
            clustered_data[chunk_index, :, cluster_index] = np.max(segmentation[:, cluster == cluster_index], axis=1)

    clustered_segmentations = type(segmentations)(clustered_data, segmentations.sliding_window)
    return pipeline.to_diarization(clustered_segmentations, count)


def _silent_embedding(pipeline: Any, num_frames: int, num_samples: int) -> np.ndarray:
    """Embedding the model returns for an all-zero weight mask.

    Statistics pooling with zero weights removes every dependency on the
    waveform, so this vector is a model constant.
    """
    import torch

    return pipeline._embedding(
        torch.zeros(1, 1, num_samples),
        masks=torch.zeros(1, num_frames),
    )[0]


def _get_embeddings(
    pipeline: Any,
    file: Any,
    binary_segmentations: Any,
    exclude_overlap: bool = False,
    hook: Any = None,
) -> np.ndarray:
    """Extract active pair embeddings; reuse the zero-mask constant for inactive pairs.

    Compatible backends share frame extraction across speakers. Different batch
    shapes can introduce small floating-point differences in the embeddings.
    """
    import torch

    embedding = pipeline._embedding
    duration = binary_segmentations.sliding_window.duration
    num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

    if exclude_overlap:
        # A pair needs this many frames of non-overlapping speech before the
        # clean mask is preferred over the full one.
        min_num_samples = embedding.min_num_samples
        min_num_frames = math.ceil(num_frames * min_num_samples / (duration * embedding.sample_rate))
        clean_frames = 1.0 * (np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2)
        clean_data = binary_segmentations.data * clean_frames
    else:
        min_num_frames = -1
        clean_data = binary_segmentations.data

    # Same test the pipeline uses to force-assign pairs to the throw-away cluster.
    active = np.sum(binary_segmentations.data, axis=1) != 0

    num_samples = round(duration * embedding.sample_rate)
    embeddings = np.tile(
        _silent_embedding(pipeline, num_frames, num_samples),
        (num_chunks, num_speakers, 1),
    )

    def iter_active_pairs():
        for chunk_index, (chunk, masks) in enumerate(binary_segmentations):
            speaker_indices = np.flatnonzero(active[chunk_index])
            if not len(speaker_indices):
                continue
            waveform, _ = pipeline._audio.crop(file, chunk, mode="pad")
            # Masks may contain NaN where a chunk was only partially stitched.
            masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
            clean_masks = np.nan_to_num(clean_data[chunk_index], nan=0.0).astype(np.float32)
            for speaker_index in speaker_indices:
                clean_mask = clean_masks[:, speaker_index]
                used_mask = clean_mask if np.sum(clean_mask) > min_num_frames else masks[:, speaker_index]
                yield chunk_index, speaker_index, waveform[None], torch.from_numpy(used_mask)[None]

    batch_size = pipeline.embedding_batch_size
    batch_chunks = getattr(pipeline, "_batch_embedding_chunks", False)
    batch_count = math.ceil(int(np.any(active, axis=1).sum() if batch_chunks else active.sum()) / batch_size)
    if hook is not None:
        hook("embeddings", None, total=batch_count, completed=0)

    batch: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
    completed = 0

    def flush() -> None:
        nonlocal completed
        masks = torch.vstack([mask for _, _, _, mask in batch])
        if getattr(pipeline, "_share_embedding_frames", False):
            embedding_batch = _shared_embedding_batch(embedding, batch, masks)
        else:
            embedding_batch = embedding(
                torch.vstack([waveform for _, _, waveform, _ in batch]),
                masks=masks,
            )
        for (chunk_index, speaker_index, _, _), vector in zip(batch, embedding_batch):
            embeddings[chunk_index, speaker_index] = vector
        completed += 1
        if hook is not None:
            hook("embeddings", embedding_batch, total=batch_count, completed=completed)
        batch.clear()

    for pair in iter_active_pairs():
        if batch_chunks and batch:
            if pair[0] != batch[-1][0] and len({item[0] for item in batch}) == batch_size:
                flush()
        batch.append(pair)
        if not batch_chunks and len(batch) == batch_size:
            flush()
    if batch:
        flush()

    return embeddings


def _supports_shared_frames(embedding: Any) -> bool:
    """Check pyannote's split-frame contract without importing a model family.

    The split API extracts mask-independent frames and then applies weighted
    pooling. Unknown preprocessing settings and training mode keep the normal
    embedding path. This is internal diarization work, not an identity embedding
    API or a choice of downstream speaker-recognition model.
    """
    model = getattr(embedding, "model_", None)
    settings = getattr(model, "hparams", None)
    get_setting = getattr(settings, "get", None)
    return (
        getattr(model, "training", True) is False
        and callable(get_setting)
        and get_setting("dither") == 0.0
        and callable(getattr(model, "forward_frames", None))
        and callable(getattr(model, "forward_embedding", None))
    )


def install(pipeline: Any) -> None:
    """Replace two pyannote pipeline steps with equivalent, cheaper versions."""
    for name in ("reconstruct", "get_embeddings"):
        if not callable(getattr(pipeline, name, None)):
            raise RuntimeError(f"The pyannote diarization pipeline no longer provides {name}().")
    share_embedding_frames = _supports_shared_frames(getattr(pipeline, "_embedding", None))
    pipeline._share_embedding_frames = share_embedding_frames
    pipeline._batch_embedding_chunks = share_embedding_frames
    pipeline.reconstruct = MethodType(_reconstruct, pipeline)
    pipeline.get_embeddings = MethodType(_get_embeddings, pipeline)
