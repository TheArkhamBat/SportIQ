import logging
import os
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logger = logging.getLogger('sportiq')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def create_directories():
    """Create necessary directories"""
    dirs = ['models', 'data/raw', 'data/processed', 'logs', 'results/exports']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    return True
