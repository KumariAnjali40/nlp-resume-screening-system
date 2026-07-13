# src/utils.py
import logging
import os
import random
import numpy as np

# Configure system-wide logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ResumeScreener")

def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility across numpy, random, and scikit-learn."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed globally set to {seed}")

def ensure_directories(dirs: list):
    """Ensures that the required project directories exist."""
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info(f"Created directory: {d}")