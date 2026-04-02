import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DatasetPreprocessor:
    """
    Convert raw multi-source downloads into a clean,
    paired grayscale <-> color dataset.

    Pipeline:
    1. Validate images (corrupt, too small, already grayscale)
    2. Center-crop to square
    3. Resize to target resolution
    4. Create grayscale version
    5. Split into train / val / test
    6. Generate metadata CSV
    7. Generate attribution files
    """

    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        target_size: int = 256,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.metadata = []
        self.errors = []

    # Validation
    def _is_valid_image(self, img_path: Path) -> tuple:
        """
        Validate image quality.

        Returns:
            (is_valid: bool, reason: str)
        """
        try:
            img = Image.open(img_path)
            img.verify()

            # Reopen after verify
            img = Image.open(img_path)
            w, h = img.size

            # Too small
            if w < 200 or h < 200:
                return False, "too_small"

            # Already grayscale
            if img.mode == "L":
                return False, "grayscale"

            # Unsupported mode
            if img.mode not in ("RGB", "RGBA"):
                return False, f"bad_mode_{img.mode}"

            # Check if nearly grayscale (low saturation)
            img_rgb = img.convert("RGB")
            img_np = np.array(img_rgb)
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            mean_sat = hsv[:, :, 1].mean()
            if mean_sat < 15:
                return False, "low_saturation"

            # Check if too dark or too bright
            mean_val = hsv[:, :, 2].mean()
            if mean_val < 20:
                return False, "too_dark"
            if mean_val > 250:
                return False, "too_bright"

            return True, "ok"

        except Exception as e:
            return False, f"error_{str(e)[:50]}"

    # Processing
    def _process_single(
        self,
        img_path: Path,
        idx: int,
        split: str,
        category: str,
        source: str,
    ) -> Optional[dict]:
        """Process one image: crop, resize, create grayscale pair."""
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)

            h, w = img_np.shape[:2]

            # Center crop to square
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            cropped = img_np[
                top : top + min_dim,
                left : left + min_dim,
            ]

            # Resize
            size = (self.target_size, self.target_size)
            color = cv2.resize(
                cropped,
                size,
                interpolation=cv2.INTER_LANCZOS4,
            )

            # Create grayscale
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)

            # File name
            fname = f"{category}_{idx:06d}.jpg"

            # Save paths
            color_path = self.output_dir / split / "color" / fname
            gray_path = self.output_dir / split / "grayscale" / fname

            # Save color (RGB → BGR for cv2)
            cv2.imwrite(
                str(color_path),
                cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            # Save grayscale
            cv2.imwrite(
                str(gray_path),
                gray,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            return {
                "filename": fname,
                "split": split,
                "category": category,
                "source": source,
                "original": img_path.name,
                "height": self.target_size,
                "width": self.target_size,
            }

        except Exception as e:
            self.errors.append(f"{img_path}: {e}")
            return None

    # Main Pipeline
    def process(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        """
        Full preprocessing pipeline.

        Args:
            val_ratio:  Fraction for validation set
            test_ratio: Fraction for test set
        """
        print(f"\n{'═' * 60}")
        print(f"  🔧 PREPROCESSING PIPELINE")
        print(f"  Raw dir:  {self.raw_dir}")
        print(f"  Output:   {self.output_dir}")
        print(f"  Size:     {self.target_size}×{self.target_size}")
        print(
            f"  Split:    train={1-val_ratio-test_ratio:.0%} │ "
            f"val={val_ratio:.0%} │ test={test_ratio:.0%}"
        )
        print(f"{'═' * 60}\n")

        # Create output directories
        for split in ["train", "val", "test"]:
            for img_type in ["color", "grayscale"]:
                d = self.output_dir / split / img_type
                d.mkdir(parents=True, exist_ok=True)

        # Gather images by category
        extensions = {".jpg", ".jpeg", ".png"}
        global_idx = 0
        category_stats = {}

        for cat_dir in sorted(self.raw_dir.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name.startswith("."):
                continue

            category = cat_dir.name
            print(f"  📁 {category}")

            # Collect all images from all sources
            all_images = []
            for source_dir in sorted(cat_dir.iterdir()):
                if not source_dir.is_dir():
                    continue
                source = source_dir.name

                imgs = [
                    (p, source)
                    for p in source_dir.iterdir()
                    if p.suffix.lower() in extensions and not p.name.startswith(".")
                ]
                all_images.extend(imgs)
                print(f"     └─ {source}: {len(imgs):,} files")

            if not all_images:
                continue

            # Step 1: Validate
            valid_images = []
            reject_reasons = {}
            for img_path, source in tqdm(
                all_images,
                desc=f"     Validating",
                leave=False,
            ):
                ok, reason = self._is_valid_image(img_path)
                if ok:
                    valid_images.append((img_path, source))
                else:
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

            print(f"     ✅ Valid: {len(valid_images):,} / " f"{len(all_images):,}")
            if reject_reasons:
                top_reasons = sorted(
                    reject_reasons.items(),
                    key=lambda x: -x[1],
                )[:3]
                for reason, count in top_reasons:
                    print(f"        ❌ {reason}: {count}")

            if not valid_images:
                continue

            # Step 2: Shuffle and split
            np.random.seed(42)
            indices = np.arange(len(valid_images))
            np.random.shuffle(indices)

            n = len(valid_images)
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            n_train = n - n_val - n_test

            splits = {}
            splits["train"] = indices[:n_train]
            splits["val"] = indices[n_train : n_train + n_val]
            splits["test"] = indices[n_train + n_val :]

            # Step 3: Process each split
            cat_count = 0
            for split_name, split_indices in splits.items():
                for idx_in_split in tqdm(
                    split_indices,
                    desc=f"     {split_name:5s}",
                    leave=False,
                ):
                    img_path, source = valid_images[idx_in_split]

                    result = self._process_single(
                        img_path,
                        global_idx,
                        split_name,
                        category,
                        source,
                    )

                    if result:
                        self.metadata.append(result)
                        global_idx += 1
                        cat_count += 1

            category_stats[category] = cat_count
            print(f"     📊 Processed: {cat_count:,}\n")

        # Step 4: Save metadata
        df = pd.DataFrame(self.metadata)
        df.to_csv(
            self.output_dir / "metadata.csv",
            index=False,
        )

        # Step 5: Collect attribution files
        self._collect_attributions()

        # Step 6: Create license file
        self._create_license_files()

        # Step 7: Print summary
        self._print_summary(df, category_stats)

        return df

    def _collect_attributions(self):
        """
        Merge all per-source ATTRIBUTION.csv files
        into one master file.
        """
        master_attr = self.output_dir / "ATTRIBUTION.csv"
        all_rows = []

        for attr_file in self.raw_dir.rglob("ATTRIBUTION.csv"):
            try:
                cat = attr_file.parent.parent.name
                source = attr_file.parent.name

                with open(attr_file) as f:
                    import csv

                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        all_rows.append([cat, source] + row)
            except Exception:
                pass

        if all_rows:
            with open(master_attr, "w", newline="") as f:
                import csv

                writer = csv.writer(f)
                writer.writerow(
                    [
                        "category",
                        "source",
                        "filename",
                        "id",
                        "creator",
                        "license",
                        "url",
                    ]
                )
                writer.writerows(all_rows)

            print(f"\n  📜 Attribution: {len(all_rows):,} entries")

    def _create_license_files(self):
        """Create LICENSE and SOURCE_LICENSES.md."""

        # Main license
        license_text = """
Creative Commons Attribution-ShareAlike 4.0
International (CC-BY-SA 4.0)

This dataset is a compilation of images from
multiple openly-licensed sources.

See SOURCE_LICENSES.md for per-source licensing.
See ATTRIBUTION.csv for per-image attribution.

https://creativecommons.org/licenses/by-sa/4.0/
        """.strip()

        (self.output_dir / "LICENSE").write_text(license_text)

        # Source licenses
        source_md = """# Data Sources & Licenses

| Source | License | URL |
|--------|---------|-----|
| COCO 2017 | CC-BY 4.0 | https://cocodataset.org |
| Open Images V7 | CC-BY 4.0 | https://storage.googleapis.com/openimages |
| Places365 | MIT Academic | http://places2.csail.mit.edu |
| iNaturalist | CC0/CC-BY/CC-BY-SA (per-photo) | https://www.inaturalist.org |
| WikiMedia Commons | CC-BY-SA/CC0/PD (per-file) | https://commons.wikimedia.org |

## Individual Attribution
Per-image attribution is provided in `ATTRIBUTION.csv`.

## Dataset License
This compiled dataset is released under **CC-BY-SA 4.0**.
        """.strip()

        (self.output_dir / "SOURCE_LICENSES.md").write_text(source_md)

    def _print_summary(
        self,
        df: pd.DataFrame,
        category_stats: dict,
    ):
        """LLM Generated: Print final summary report."""
        print(f"\n{'═' * 60}")
        print(f"  📊 DATASET SUMMARY")
        print(f"{'═' * 60}")
        print(f"\n  Total images: {len(df):,}")
        print(f"  Resolution:   " f"{self.target_size}×{self.target_size}")

        print(f"\n  Per split:")
        for split, count in df["split"].value_counts().items():
            pct = count / len(df) * 100
            print(f"    {split:5s}: {count:>6,} ({pct:.1f}%)")

        print(f"\n  Per category:")
        for cat, count in sorted(
            category_stats.items(),
            key=lambda x: -x[1],
        ):
            print(f"    {cat:12s}: {count:>6,}")

        print(f"\n  Per source:")
        for source, count in df["source"].value_counts().items():
            print(f"    {source:15s}: {count:>6,}")

        print(f"\n  Output: {self.output_dir}")
        print(f"\n  Files:")
        print(f"    📄 metadata.csv")
        print(f"    📄 LICENSE")
        print(f"    📄 SOURCE_LICENSES.md")
        print(f"    📄 ATTRIBUTION.csv")

        if self.errors:
            print(f"\n  ⚠️  {len(self.errors)} processing errors")

        print(f"{'═' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess raw images into " "paired colorization dataset"
    )
    parser.add_argument(
        "--raw-dir",
        "-r",
        type=Path,
        default=Path("raw_downloads"),
        help="Raw downloads directory",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("colorization_dataset_v1"),
        help="Output dataset directory",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=256,
        help="Target image size (default: 256)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    processor = DatasetPreprocessor(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        target_size=args.size,
    )
    processor.process(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
