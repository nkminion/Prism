import json
import random
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

import config
from tqdm import tqdm

from .base import BaseScraper


class COCOScraper(BaseScraper):
    """
    COCO 2017 - CC-BY 4.0 licensed.
    330K images, 80 object categories.

    Downloads the full dataset ONCE (~19GB), then filters
    by category for each request. Very fast after initial DL.

    License: https://cocodataset.org/#termsofuse
    """

    URLS = {
        "train_images": ("http://images.cocodataset.org/zips/train2017.zip"),
        "val_images": ("http://images.cocodataset.org/zips/val2017.zip"),
        "annotations": (
            "http://images.cocodataset.org/annotations/" "annotations_trainval2017.zip"
        ),
    }

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, "coco")
        self.coco_dir = Path(config.COCO_CACHE)
        self.coco_dir.mkdir(parents=True, exist_ok=True)
        self.annotations = None
        self.cat_name_to_id = {}
        self.cat_to_images = defaultdict(set)
        self.imgid_to_filename = {}

    # Setup
    def _ensure_downloaded(self):
        """Download COCO 2017 if not present."""
        ann_file = self.coco_dir / "annotations" / "instances_train2017.json"

        # Download annotations
        if not ann_file.exists():
            self.logger.info("Downloading COCO annotations...")
            zip_path = self.coco_dir / "annotations.zip"
            subprocess.run(
                [
                    "wget",
                    "-c",
                    "-q",
                    "--show-progress",
                    "-O",
                    str(zip_path),
                    self.URLS["annotations"],
                ],
                check=True,
            )
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.coco_dir)
            zip_path.unlink()
        else:
            self.logger.info("COCO annotations cached")

        # Download train images
        img_dir = self.coco_dir / "train2017"
        if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) > 100_000:
            self.logger.info("COCO train images cached")
        else:
            self.logger.info("Downloading COCO train images (~18GB)...")
            zip_path = self.coco_dir / "train2017.zip"
            subprocess.run(
                [
                    "wget",
                    "-c",
                    "-q",
                    "--show-progress",
                    "-O",
                    str(zip_path),
                    self.URLS["train_images"],
                ],
                check=True,
            )
            self.logger.info("Extracting COCO images...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.coco_dir)

    def _load_annotations(self):
        """Load COCO annotations into memory (once)."""
        if self.annotations is not None:
            return

        ann_file = self.coco_dir / "annotations" / "instances_train2017.json"
        self.logger.info("Loading COCO annotations...")
        with open(ann_file, "r") as f:
            self.annotations = json.load(f)

        self.cat_name_to_id = {
            c["name"]: c["id"] for c in self.annotations["categories"]
        }

        self.cat_to_images = defaultdict(set)
        for ann in self.annotations["annotations"]:
            self.cat_to_images[ann["category_id"]].add(ann["image_id"])

        self.imgid_to_filename = {
            img["id"]: img["file_name"] for img in self.annotations["images"]
        }

        self.logger.info(
            f"COCO: {len(self.annotations['images']):,} images, "
            f"{len(self.annotations['categories'])} categories"
        )

    def scrape(
        self,
        category: str,
        coco_classes: list,
        target_count: int,
    ) -> int:
        """
        Extract images containing specified COCO classes.

        Args:
            category:     output category name
            coco_classes: COCO class names, e.g. ["car", "truck"]
            target_count: images to extract
        """
        if not coco_classes:
            return 0

        self.logger.info(
            f"COCO: {category} " f"(classes={coco_classes}, target={target_count})"
        )

        self._ensure_downloaded()
        self._load_annotations()

        # Find matching image IDs
        all_image_ids = set()
        for cls_name in coco_classes:
            if cls_name not in self.cat_name_to_id:
                self.logger.warning(
                    f"  Unknown COCO class: '{cls_name}'. "
                    f"Available: {sorted(self.cat_name_to_id.keys())}"
                )
                continue
            cat_id = self.cat_name_to_id[cls_name]
            ids = self.cat_to_images[cat_id]
            self.logger.info(f"  {cls_name}: {len(ids):,} images")
            all_image_ids.update(ids)

        self.logger.info(f"  Total unique: {len(all_image_ids):,}")

        # Check existing
        existing = self.get_existing_count(category)
        remaining = target_count - existing
        if remaining <= 0:
            return existing

        # Sample and copy
        candidates = list(all_image_ids)
        random.shuffle(candidates)
        selected = candidates[:remaining]

        save_dir = self.output_dir / category / self.source_name
        save_dir.mkdir(parents=True, exist_ok=True)
        copied = 0

        for img_id in tqdm(
            selected,
            desc=f"    Copying {category}",
            leave=False,
        ):
            filename = self.imgid_to_filename.get(img_id)
            if filename is None:
                continue

            src = self.coco_dir / "train2017" / filename
            dst = save_dir / f"coco_{filename}"

            if dst.exists():
                copied += 1
                continue

            if src.exists():
                shutil.copy2(src, dst)
                copied += 1

        total = existing + copied
        self.logger.info(f"✅ COCO {category}: {copied} copied, {total} total")
        return total
