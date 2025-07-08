import uuid


def generate_random_name(suffix=".mp4"):
    """Generates random name."""
    return f"{uuid.uuid4()}{suffix}"
