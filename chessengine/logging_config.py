import logging
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "chess_engine.log")

def setup_logger():
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)