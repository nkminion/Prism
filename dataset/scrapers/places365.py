import random
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Optional

import config
from tqdm import tqdm

from .base import BaseScraper


class Places365Scraper(BaseScraper):
    """
    Places365-Standard - MIT academic license.
    1.8M images across 365 scene categories.

    Explicitly built for ML/vision research.
    License: http://places2.csail.mit.edu/download.html

    Downloads once (~25GB), then filters by scene category.
    """

    DOWNLOAD_URL = (
        "http://data.csail.mit.edu/places/places365/" "places365standard_easyformat.tar"
    )

    CATEGORIES_URL = (
        "https://raw.githubusercontent.com/CSAILVision/"
        "places365/master/categories_places365.txt"
    )

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, "places365")
        self.cache_dir = Path(config.PLACES_CACHE)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.category_map: Optional[dict[str, str]] = None

    # Setup
    def _ensure_downloaded(self):
        """Download Places365 (one-time, ~25GB)."""
        # Check if already extracted
        train_dir = self.cache_dir / "places365_standard" / "train"
        if not train_dir.exists():
            # Also check alternate extraction path
            train_dir = self.cache_dir / "data_256" / "train"

        if train_dir.exists():
            count = sum(1 for _ in train_dir.rglob("*.jpg"))
            if count > 100_000:
                self.logger.info(f"Places365 cached ({count:,} images)")
                self._train_dir = train_dir
                return

        self.logger.info("Downloading Places365-Standard (~25GB)...")
        tar_path = self.cache_dir / "places365.tar"

        subprocess.run(
            [
                "wget",
                "-c",
                "-q",
                "--show-progress",
                "-O",
                str(tar_path),
                self.DOWNLOAD_URL,
            ],
            check=True,
        )

        self.logger.info("Extracting Places365...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(self.cache_dir)

        # Find the train directory
        for candidate in [
            self.cache_dir / "places365_standard" / "train",
            self.cache_dir / "data_256" / "train",
            self.cache_dir / "train",
        ]:
            if candidate.exists():
                self._train_dir = candidate
                break
        else:
            # Search for it
            for d in self.cache_dir.rglob("train"):
                if d.is_dir():
                    self._train_dir = d
                    break

        self.logger.info("Places365 ready")

    def _load_categories(self):
        """Load 365 scene category names."""
        if self.category_map is not None:
            return

        cat_file = self.cache_dir / "categories_places365.txt"
        if not cat_file.exists():
            import requests

            resp = requests.get(self.CATEGORIES_URL)
            cat_file.write_text(resp.text)

        self.category_map = {}
        with open(cat_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    cat_path = parts[0]  # e.g., /a/abbey
                    cat_name = cat_path.split("/")[-1]
                    self.category_map[cat_name] = cat_path

        self.logger.info(f"Loaded {len(self.category_map)} scene categories")

    def _find_matching(self, keywords: list) -> list:
        """Find Places365 categories matching keywords."""
        if self.category_map is None:
            self._load_categories()

        matches = []
        seen = set()

        for keyword in keywords:
            kw = keyword.lower().replace(" ", "_").replace("/", "_")
            for cat_name, cat_path in self.category_map.items():  # type: ignore[ty:unresolved-attribute]
                cn = cat_name.lower()
                if kw in cn or cn in kw:
                    if cat_name not in seen:
                        seen.add(cat_name)
                        matches.append((cat_name, cat_path))

        return matches

    def scrape(
        self,
        category: str,
        places_categories: list,
        target_count: int,
    ) -> int:
        """
        Extract images from Places365 by scene category.
        """
        if not places_categories:
            return 0

        if self.category_map is None:
            self._load_categories()

        self.logger.info(
            f"Places365: {category} "
            f"(scenes={places_categories}, target={target_count})"
        )

        self._ensure_downloaded()
        self._load_categories()

        existing = self.get_existing_count(category)
        remaining = target_count - existing
        if remaining <= 0:
            return existing

        # Match keywords to actual categories
        matched = self._find_matching(places_categories)
        if not matched:
            self.logger.warning(f"  No matches for: {places_categories}")
            self.logger.info(f"  Available: {sorted(self.category_map.keys())[:20]}...")  # type: ignore[ty:unresolved-attribute]
            return existing

        self.logger.info(
            f"  Matched {len(matched)} categories: "
            f"{[m[0] for m in matched[:10]]}..."
        )

        per_cat = remaining // len(matched) + 1
        save_dir = self.output_dir / category / self.source_name
        save_dir.mkdir(parents=True, exist_ok=True)
        copied = 0

        for cat_name, cat_path in matched:
            if copied >= remaining:
                break

            src_dir = self._train_dir / cat_path.lstrip("/")
            if not src_dir.exists():
                # Try without leading letter directory
                src_dir = self._train_dir / cat_name
            if not src_dir.exists():
                self.logger.debug(f"  Dir not found: {src_dir}")
                continue

            images = sorted(src_dir.glob("*.jpg"))
            random.shuffle(images)
            images = images[:per_cat]

            for img_path in tqdm(
                images,
                desc=f"    {cat_name}",
                leave=False,
            ):
                if copied >= remaining:
                    break

                dst = save_dir / f"places_{cat_name}_{img_path.name}"
                if not dst.exists():
                    try:
                        shutil.copy2(img_path, dst)
                        copied += 1
                    except Exception as e:
                        self.logger.debug(f"Copy failed: {e}")

        total = existing + copied
        self.logger.info(f"✅ Places365 {category}: " f"{copied} copied, {total} total")
        return total
