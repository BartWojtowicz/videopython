import logging
import os

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")


def _setup_logger():
    logger = logging.getLogger()

    logging_level = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
    }.get(LOG_LEVEL.lower())

    logging_format = "[{asctime}|{filename}:{funcName}:{lineno:d}]{levelname}  {message}"
    logging.basicConfig(format=logging_format, style="{", level=logging_level)
    formatter = logging.Formatter(logging_format, style="{", datefmt="%H:%M:%S")

    # Create a handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add our console handler to the logger
    logger.removeHandler(logger.handlers[0])
    logger.addHandler(console_handler)

    return logger


logger = _setup_logger()
