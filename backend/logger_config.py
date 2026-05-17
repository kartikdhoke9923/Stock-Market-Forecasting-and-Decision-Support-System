"""
logger_config.py — Centralized logging setup
Logs to both console (CloudWatch auto-captures) and file.
"""
import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str = "stock_analyzer") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — AWS CloudWatch captures stdout automatically
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler — local log file (useful for debugging)
    try:
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass  # File logging optional — don't crash if no write permission

    # Silence noisy third-party libraries
    for noisy in ["yfinance","urllib3","httpx","hpack","h2","httpcore","peewee"]:
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    return logger


# Single shared logger instance
log = setup_logger()
