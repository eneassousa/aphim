import logging
import os
import sys
from pathlib import Path

def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    
    # File Handler
    f_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    
    return logger
