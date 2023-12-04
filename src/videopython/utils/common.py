import time
import uuid


def generate_random_name(suffix=".mp4"):
    """Generates random video name."""
    return f"{uuid.uuid4()}{suffix}"


def timeit(func: callable):
    """Decorator to measure execution time of a function."""

    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start:.3f} seconds.")
        return result

    return timed
