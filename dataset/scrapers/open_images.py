import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
from tqdm import tqdm

from .base import BaseScraper


class OpenImagesScraper(BaseScraper):
    """
    Open Images V7 - CC-BY 4.0 licensed.
    9M+ images with labels. Bulk download from AWS S3.

    License: https://storage.googleapis.com/openimages/web/factsfigures.html
    Explicitly designed for ML research.
    """

    METADATA_BASE = "https://storage.googleapis.com/openimages/v7"
    TRAIN_LABELS_URL = (
        f"{METADATA_BASE}/" f"oidv7-train-annotations-human-imagelabels.csv"
    )
    CLASS_DESC_URL = f"{METADATA_BASE}/oidv7-class-descriptions.csv"

    # Open Images stores images on AWS S3 with predictable URLs:
    IMAGE_URL_TEMPLATE = "https://s3.amazonaws.com/open-images-dataset/train/{}.jpg"

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, "open_images")
        self.metadata_dir = Path(config.OI_CACHE)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self._labels_loaded = False
        self._image_ids_by_label = {}

    # Metadata
    def _download_metadata(self):
        """Download label CSVs (one-time, ~500MB)."""
        files = {
            "class_descriptions.csv": self.CLASS_DESC_URL,
            "train_labels.csv": self.TRAIN_LABELS_URL,
        }
        for fname, url in files.items():
            fpath = self.metadata_dir / fname
            if fpath.exists():
                self.logger.info(f"Metadata cached: {fname}")
                continue

            self.logger.info(f"Downloading {fname}...")
            subprocess.run(
                ["wget", "-q", "--show-progress", "-O", str(fpath), url],
                check=True,
            )

    def _load_labels(self):
        """Parse label CSV into memory (once)."""
        if self._labels_loaded:
            return

        self._download_metadata()

        labels_file = self.metadata_dir / "train_labels.csv"
        self.logger.info("Loading Open Images labels...")

        self._image_ids_by_label = {}

        with open(labels_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["LabelName"]
                conf = float(row.get("Confidence", 1))
                if conf < 0.8:
                    continue

                if label not in self._image_ids_by_label:
                    self._image_ids_by_label[label] = []
                self._image_ids_by_label[label].append(row["ImageID"])

        total = sum(len(v) for v in self._image_ids_by_label.values())
        self.logger.info(
            f"Loaded {total:,} label-image pairs "
            f"across {len(self._image_ids_by_label)} labels"
        )
        self._labels_loaded = True

    # Download
    def _download_batch(
        self,
        image_ids: list,
        save_dir: Path,
        max_workers: int = 16,
    ) -> int:
        """Parallel download from S3."""
        save_dir.mkdir(parents=True, exist_ok=True)
        success = 0

        def _dl_one(img_id):
            save_path = save_dir / f"oi_{img_id}.jpg"
            if save_path.exists():
                return True
            url = self.IMAGE_URL_TEMPLATE.format(img_id)
            return self.download_image(url, save_path)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_dl_one, iid): iid for iid in image_ids}
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"    DL ({save_dir.parent.name})",
                leave=False,
            )
            for fut in pbar:
                iid = futures[fut]
                try:
                    if fut.result():
                        success += 1
                        self._mark_downloaded(iid)
                except Exception:
                    pass
                pbar.set_postfix(ok=success)

        return success

    # Main Entry
    def scrape(
        self,
        category: str,
        label_codes: list,
        target_count: int,
    ) -> int:
        """
        Download Open Images photos matching given label codes.

        Args:
            category:     e.g. "humans"
            label_codes:  e.g. ["/m/01g317", "/m/0dzct"]
            target_count: how many images to download
        """
        self.logger.info(
            f"Open Images: {category} " f"(labels={label_codes}, target={target_count})"
        )

        # Check existing
        existing = self.get_existing_count(category)
        remaining = target_count - existing
        if remaining <= 0:
            self.logger.info(f"Already have {existing} images. Skipping.")
            return existing

        # Load label metadata
        self._load_labels()

        # Collect candidate image IDs across all requested labels
        candidate_ids = set()
        for label in label_codes:
            ids = self._image_ids_by_label.get(label, [])
            self.logger.info(f"  Label {label}: {len(ids):,} candidates")
            candidate_ids.update(ids)

        # Remove already-downloaded
        candidate_ids -= self.downloaded_ids

        # Sample what we need + buffer for failures
        import random

        candidates = list(candidate_ids)
        random.shuffle(candidates)
        to_download = candidates[: int(remaining * 1.3)]

        self.logger.info(
            f"  Unique candidates: {len(candidate_ids):,}, "
            f"attempting: {len(to_download):,}"
        )

        # Download
        save_dir = self.output_dir / category / self.source_name
        downloaded = self._download_batch(to_download, save_dir)

        total = existing + downloaded
        self.logger.info(
            f"✅ Open Images {category}: " f"{downloaded} new, {total} total"
        )
        return total
