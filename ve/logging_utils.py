from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(output_dir: str | Path, level: str = "INFO", filename: str = "pipeline.log") -> None:
    """Configure Loguru sinks for console and file under output_dir/logs.

    - Console: colored, level `level`
    - File: output_dir/logs/filename with rotation and retention
    """
    out = Path(output_dir)
    logs_dir = out / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Remove any default handlers to avoid duplicate logs
    logger.remove()

    # Console sink
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level.upper(),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n",
    )

    # File sink
    logger.add(
        logs_dir / filename,
        level=level.upper(),
        rotation="10 MB",
        retention=10,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
