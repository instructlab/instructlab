# Standard
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def lookup_processed_documents_folder(output_dir: str) -> Optional[Path]:
    latest_folder = (
        max(Path(output_dir).iterdir(), key=lambda d: d.stat().st_mtime)
        if Path(output_dir).exists()
        else None
    )
    logger.info(f"Latest processed folder is {latest_folder}")

    if latest_folder is not None:
        docling_folder = Path.joinpath(latest_folder, "docling-artifacts")
        logger.debug(f"Docling folder: {docling_folder}")
        if docling_folder.exists():
            return docling_folder
    return None
