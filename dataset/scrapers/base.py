import logging
import time
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-14s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


class BaseScraper(ABC):
    """Abstract base class for all image scrapers / extractors."""

    def __init__(self, output_dir: Path, source_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_name = source_name
        self.logger = logging.getLogger(source_name)

        # HTTP session with retries
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=20,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "User-Agent": (
                    "PrismDatasetBot/1.0 "
                    "(open-source ML research dataset; "
                    "https://kaggle.com/datasets/pythonicvarun/colorization-dataset)"
                )
            }
        )

        # Resume support: track what's already downloaded
        self.downloaded_ids = set()
        self.log_file = self.output_dir / f".{source_name}_progress.log"
        self._load_progress()

    def _load_progress(self):
        """Load previously downloaded IDs for resume."""
        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                self.downloaded_ids = {line.strip() for line in f if line.strip()}
            if self.downloaded_ids:
                self.logger.info(
                    f"Resume: {len(self.downloaded_ids)} " f"already downloaded"
                )

    def _mark_downloaded(self, image_id: str):
        """Record a successful download for resume."""
        self.downloaded_ids.add(str(image_id))
        with open(self.log_file, "a") as f:
            f.write(f"{image_id}\n")

    def download_image(
        self,
        url: str,
        save_path: Path,
        min_size: int = 256,
        timeout: int = 20,
    ) -> bool:
        """
        Download a single image with validation.

        Checks:
        - HTTP success
        - Valid image file (can be opened by PIL)
        - Minimum resolution (width & height >= min_size)
        - Supported color mode (RGB, RGBA, L)
        """
        try:
            resp = self.session.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()

            content = resp.content
            if len(content) < 1000:  # Too small to be a real image
                return False

            # Validate image
            img = Image.open(BytesIO(content))
            w, h = img.size

            if w < min_size or h < min_size:
                return False

            if img.mode not in ("RGB", "RGBA", "L"):
                return False

            # Save
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(content)

            return True

        except (
            requests.RequestException,
            IOError,
            OSError,
            Image.DecompressionBombError,
        ) as e:
            self.logger.debug(f"Download failed {url}: {e}")
            return False

    def get_existing_count(self, category: str) -> int:
        """Count images already downloaded for a category+source."""
        cat_dir = self.output_dir / category / self.source_name
        if not cat_dir.exists():
            return 0
        count = (
            len(list(cat_dir.glob("*.jpg")))
            + len(list(cat_dir.glob("*.jpeg")))
            + len(list(cat_dir.glob("*.png")))
        )
        return count

    @abstractmethod
    def scrape(self, category: str, identifiers, target_count: int) -> int:
        """
        Download/extract images for a category.

        Args:
            category:     Output category name (e.g. "animals")
            identifiers:  Source-specific IDs (labels, classes, taxa, etc.)
            target_count: Number of images to collect

        Returns:
            Total number of images available after this run
        """
        pass
