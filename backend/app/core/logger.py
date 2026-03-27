"""Centralized logging configuration."""

import logging
import sys


def setup_logger(name: str = "traffic_monitor") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


logger = setup_logger()
