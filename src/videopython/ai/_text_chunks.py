import re


def split_text(text: str, max_chars: int) -> list[str]:
    """Prefer sentence, then clause, then word boundaries; retain every character.

    Whitespace is normalized. Text without usable boundaries (including scripts
    without spaces and long URLs) is split at the character budget as a last resort.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    remaining = " ".join(text.split())
    chunks: list[str] = []
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        boundary = 0
        for pattern in (r"(?:[.!?](?:\s|$)|[。！？])", r"(?:[,;:](?:\s|$)|[，；：])", r"\s"):
            matches = [
                match
                for match in re.finditer(pattern, window)
                if match.start() + (0 if pattern == r"\s" else 1) <= max_chars
            ]
            if matches:
                boundary = matches[-1].start() + (0 if pattern == r"\s" else 1)
                break
        if not boundary:
            boundary = max_chars
        chunks.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks
