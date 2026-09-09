from videopython.ai.dubbing.models import TranslatedSegment

MAX_GROUP_SECONDS = 10.0
MAX_GROUP_MEMBERS = 4
MIN_SYNTHESIS_SECONDS = 0.1


def synthesis_groups(segments: list[TranslatedSegment]) -> list[list[int]]:
    groups: list[list[int]] = []
    for i, segment in enumerate(segments):
        if groups:
            previous = segments[groups[-1][-1]]
            tiny = (
                min(previous.duration, segment.duration) < 1.0
                or min(len(previous.translated_text.strip()), len(segment.translated_text.strip())) < 10
            )
            if (
                tiny
                and len(groups[-1]) < MAX_GROUP_MEMBERS
                and segment.end - segments[groups[-1][0]].start <= MAX_GROUP_SECONDS
                and previous.speaker == segment.speaker
                and 0 <= segment.start - previous.end <= 0.15
                and previous.translated_text.strip()
                and segment.translated_text.strip()
            ):
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups
