# logger.py
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Explicitly set to DEBUG
    logger.propagate = False
    formatter = logging.Formatter(
        "%(levelname)s:    [%(asctime)s] - %(name)s - %(message)s"
    )
    handler = logging.StreamHandler(sys.stdout) 
    handler.setFormatter(formatter)
    logger.addHandler(handler) 
    return logger
