"""
LLM Generated: DatasetVerifier - Verify dataset integrity before publishing.
"""

from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class DatasetVerifier:
    """
    Verify dataset integrity before publishing.

    Checks:
    1. Every color image has a matching grayscale pair
    2. No corrupt files
    3. Consistent dimensions
    4. Metadata CSV matches actual files
    5. Attribution file exists
    6. License files exist
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.issues = []

    def verify(self) -> bool:
        """Run all verification checks."""
        print(f"\n{'═' * 55}")
        print(f"  🔍 DATASET VERIFICATION")
        print(f"  {self.dataset_dir}")
        print(f"{'═' * 55}\n")

        self._check_structure()
        self._check_pairs()
        self._check_integrity()
        self._check_metadata()
        self._check_license_files()
        self._generate_preview()

        # Report
        print(f"\n{'─' * 55}")
        if self.issues:
            print(f"  ⚠️  {len(self.issues)} issues found:\n")
            for issue in self.issues[:20]:
                print(f"    • {issue}")
            if len(self.issues) > 20:
                print(f"    ... and {len(self.issues) - 20} more")
            return False
        else:
            print(f"  ✅ ALL CHECKS PASSED — Ready to publish!")
            return True

    def _check_structure(self):
        """Verify directory structure."""
        print("  📂 Checking structure...")
        required = []
        for split in ["train", "val", "test"]:
            for img_type in ["color", "grayscale"]:
                required.append(self.dataset_dir / split / img_type)

        for d in required:
            if not d.exists():
                self.issues.append(f"Missing directory: {d}")
            elif not any(d.iterdir()):
                self.issues.append(f"Empty directory: {d}")
            else:
                count = len(list(d.glob("*.jpg")))
                print(f"    {d.relative_to(self.dataset_dir)}: " f"{count:,} files")

    def _check_pairs(self):
        """Verify every color image has a grayscale pair."""
        print("\n  🔗 Checking pairs...")
        for split in ["train", "val", "test"]:
            color_dir = self.dataset_dir / split / "color"
            gray_dir = self.dataset_dir / split / "grayscale"

            if not color_dir.exists() or not gray_dir.exists():
                continue

            color_files = {f.name for f in color_dir.glob("*.jpg")}
            gray_files = {f.name for f in gray_dir.glob("*.jpg")}

            missing_gray = color_files - gray_files
            missing_color = gray_files - color_files

            if missing_gray:
                self.issues.append(
                    f"{split}: {len(missing_gray)} color images "
                    f"without grayscale pair"
                )
            if missing_color:
                self.issues.append(
                    f"{split}: {len(missing_color)} grayscale "
                    f"images without color pair"
                )

            matched = color_files & gray_files
            print(f"    {split}: {len(matched):,} matched pairs ✓")

    def _check_integrity(self):
        """Verify no corrupt files + consistent dimensions."""
        print("\n  🖼️  Checking image integrity...")
        sample_size = 500  # Check a random sample

        all_images = list(self.dataset_dir.rglob("*.jpg"))
        if len(all_images) > sample_size:
            import random

            random.seed(42)
            sample = random.sample(all_images, sample_size)
        else:
            sample = all_images

        corrupt = 0
        sizes = set()

        for img_path in tqdm(
            sample,
            desc="    Integrity check",
            leave=False,
        ):
            try:
                img = Image.open(img_path)
                img.verify()
                img = Image.open(img_path)
                sizes.add(img.size)
            except Exception:
                corrupt += 1
                self.issues.append(f"Corrupt: {img_path}")

        print(f"    Checked {len(sample):,} files, " f"{corrupt} corrupt")
        print(f"    Unique sizes: {sizes}")

        if len(sizes) > 2:  # color + grayscale may differ
            self.issues.append(f"Inconsistent dimensions: {sizes}")

    def _check_metadata(self):
        """Verify metadata.csv matches actual files."""
        print("\n  📄 Checking metadata...")
        meta_path = self.dataset_dir / "metadata.csv"

        if not meta_path.exists():
            self.issues.append("Missing metadata.csv")
            return

        df = pd.read_csv(meta_path)
        print(f"    Entries: {len(df):,}")
        print(f"    Columns: {list(df.columns)}")

        # Check required columns
        required_cols = {
            "filename",
            "split",
            "category",
        }
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            self.issues.append(f"Missing columns: {missing_cols}")

        # Spot-check that files exist
        sample = df.sample(min(100, len(df)), random_state=42)
        missing = 0
        for _, row in sample.iterrows():
            color_path = self.dataset_dir / row["split"] / "color" / row["filename"]
            if not color_path.exists():
                missing += 1

        if missing > 0:
            self.issues.append(
                f"metadata.csv references {missing} "
                f"non-existent files (of {len(sample)} checked)"
            )
        else:
            print(f"    Spot-check: {len(sample)} files verified ✓")

    def _check_license_files(self):
        """Verify license and attribution files exist."""
        print("\n  📜 Checking license files...")

        required = [
            "LICENSE",
            "SOURCE_LICENSES.md",
        ]
        for fname in required:
            fpath = self.dataset_dir / fname
            if fpath.exists():
                print(f"    ✅ {fname}")
            else:
                self.issues.append(f"Missing: {fname}")
                print(f"    ❌ {fname}")

        attr = self.dataset_dir / "ATTRIBUTION.csv"
        if attr.exists():
            lines = sum(1 for _ in open(attr)) - 1
            print(f"    ✅ ATTRIBUTION.csv ({lines:,} entries)")
        else:
            print(f"    ⚠️  ATTRIBUTION.csv (optional but recommended)")

    def _generate_preview(self):
        """Generate a visual preview grid."""
        print("\n  🖼️  Generating preview...")

        try:
            meta_path = self.dataset_dir / "metadata.csv"
            if not meta_path.exists():
                return

            df = pd.read_csv(meta_path)
            categories = sorted(df["category"].unique())

            n_cats = min(len(categories), 10)
            fig, axes = plt.subplots(
                n_cats,
                4,
                figsize=(12, 3 * n_cats),
            )

            if n_cats == 1:
                axes = axes.reshape(1, -1)

            for i, cat in enumerate(categories[:n_cats]):
                cat_df = df[(df["category"] == cat) & (df["split"] == "train")]

                if len(cat_df) < 2:
                    continue

                samples = cat_df.sample(
                    min(2, len(cat_df)),
                    random_state=42,
                )

                for j, (_, row) in enumerate(samples.iterrows()):
                    if j >= 2:
                        break

                    color_path = self.dataset_dir / "train" / "color" / row["filename"]
                    gray_path = (
                        self.dataset_dir / "train" / "grayscale" / row["filename"]
                    )

                    col_base = j * 2

                    if gray_path.exists():
                        gray_img = plt.imread(str(gray_path))
                        axes[i, col_base].imshow(
                            gray_img,
                            cmap="gray",
                        )
                        axes[i, col_base].set_title(
                            f"{cat}\n(gray)",
                            fontsize=8,
                        )
                        axes[i, col_base].axis("off")

                    if color_path.exists():
                        color_img = plt.imread(str(color_path))
                        axes[i, col_base + 1].imshow(color_img)
                        axes[i, col_base + 1].set_title(
                            f"{cat}\n(color)",
                            fontsize=8,
                        )
                        axes[i, col_base + 1].axis("off")

            plt.suptitle(
                "Dataset Preview: Grayscale → Color Pairs",
                fontsize=14,
            )
            plt.tight_layout()

            preview_path = self.dataset_dir / "preview.png"
            plt.savefig(
                str(preview_path),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            print(f"    Saved: {preview_path}")

        except Exception as e:
            print(f"    Preview generation failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify dataset before publishing")
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset directory to verify",
    )

    args = parser.parse_args()

    verifier = DatasetVerifier(args.dataset_dir)
    is_ok = verifier.verify()

    exit(0 if is_ok else 1)
