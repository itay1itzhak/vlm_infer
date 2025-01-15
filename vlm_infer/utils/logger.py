import logging
from pathlib import Path
from rich.logging import RichHandler
import sys

def setup_logger(name: str, log_file: Path, log_level: int = logging.INFO) -> logging.Logger:
    """Set up logger with console and file handlers."""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatters
    console_formatter = logging.Formatter("%(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create console handler with rich formatting
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger 