import time
import uuid


def generate_random_video_name():
    """Generates random video name."""
    return f"{uuid.uuid4()}.mp4"


def timeit(func: callable):
    """Decorator to measure execution time of a function."""

    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start:.3f} seconds.")
        return result

    return timed
