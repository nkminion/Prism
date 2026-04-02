import json
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm


class CrossSourceDeduplicator:
    """
    Remove near-duplicate images across sources.

    Uses perceptual hashing (pHash) — robust to:
    - Different resolutions
    - JPEG compression differences
    - Minor crops
    """

    def __init__(
        self,
        dataset_dir: Path,
        hash_size: int = 12,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.hash_size = hash_size

    def deduplicate(
        self,
        threshold: int = 8,
        dry_run: bool = False,
    ) -> int:
        """
        Find and remove near-duplicate images.

        Args:
            threshold: Max hamming distance for duplicates.
                       Lower = stricter.
                       6  = very strict (almost identical)
                       8  = moderate (recommended)
                       12 = loose (similar compositions)
            dry_run:   If True, report but don't delete.

        Returns:
            Number of duplicates found/removed.
        """
        print(f"\n{'═' * 55}")
        print(f"  🔍 CROSS-SOURCE DEDUPLICATION")
        print(f"  Hash size: {self.hash_size} │ " f"Threshold: {threshold}")
        print(f"{'═' * 55}\n")

        # Step 1: Collect all images
        extensions = {".jpg", ".jpeg", ".png"}
        all_images = [
            p
            for p in self.dataset_dir.rglob("*")
            if p.suffix.lower() in extensions and not p.name.startswith(".")
        ]
        print(f"  Found {len(all_images):,} images\n")

        # Step 2: Compute perceptual hashes
        hashes = {}
        errors = 0

        for img_path in tqdm(all_images, desc="  Hashing"):
            try:
                img = Image.open(img_path).convert("RGB")
                h = imagehash.phash(
                    img,
                    hash_size=self.hash_size,
                )
                hashes[img_path] = h
            except Exception:
                errors += 1

        print(f"\n  Hashed: {len(hashes):,} " f"(errors: {errors})")

        # Step 3: Group by approximate hash prefix
        buckets = defaultdict(list)
        for path, h in hashes.items():
            prefix = str(h)[:6]  # First 6 hex chars
            buckets[prefix].append((path, h))

        # Step 4: Find duplicates within buckets
        duplicates = []
        seen = set()

        for prefix, items in tqdm(
            buckets.items(),
            desc="  Comparing",
            total=len(buckets),
        ):
            for i in range(len(items)):
                if items[i][0] in seen:
                    continue
                for j in range(i + 1, len(items)):
                    if items[j][0] in seen:
                        continue

                    dist = items[i][1] - items[j][1]
                    if dist <= threshold:
                        keeper, removal = self._pick_keeper(
                            items[i][0],
                            items[j][0],
                        )
                        duplicates.append(
                            {
                                "removed": str(removal),
                                "kept": str(keeper),
                                "distance": dist,
                            }
                        )
                        seen.add(removal)

        print(f"\n  Found {len(duplicates):,} duplicates")

        # Step 5: Remove or report
        if duplicates:
            # Save report
            report_path = self.dataset_dir / "duplicates_report.json"
            with open(report_path, "w") as f:
                json.dump(duplicates, f, indent=2)
            print(f"  Report: {report_path}")

            if not dry_run:
                removed = 0
                for dup in duplicates:
                    p = Path(dup["removed"])
                    if p.exists():
                        p.unlink()
                        removed += 1

                print(f"  ✅ Removed {removed:,} duplicates")
                return removed
            else:
                print("  (Dry run — no files deleted)")
                # Show some examples
                for d in duplicates[:5]:
                    print(
                        f"    Would remove: "
                        f"{Path(d['removed']).name} "
                        f"(keep: {Path(d['kept']).name}, "
                        f"dist={d['distance']})"
                    )

        return len(duplicates)

    def _pick_keeper(
        self,
        path1: Path,
        path2: Path,
    ) -> tuple:
        """
        When two images are duplicates, pick which to keep.

        Priority (higher = keep):
          COCO > Open Images > Places365 > iNaturalist > WikiMedia

        Reasoning: COCO/OI have most consistent quality +
        are most commonly cited in papers.
        """
        priority_map = {
            "coco": 50,
            "open_images": 40,
            "places365": 30,
            "places": 30,
            "inaturalist": 20,
            "inat": 20,
            "wikimedia": 10,
            "wiki": 10,
        }

        def _score(p: Path) -> int:
            path_str = str(p).lower()
            for key, score in priority_map.items():
                if key in path_str:
                    return score
            return 0

        s1 = _score(path1)
        s2 = _score(path2)

        if s1 >= s2:
            return path1, path2
        else:
            return path2, path1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-source image deduplication")
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to deduplicate",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=8,
        help="Hamming distance threshold (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Report only, don't delete",
    )

    args = parser.parse_args()

    dedup = CrossSourceDeduplicator(args.directory)
    dedup.deduplicate(
        threshold=args.threshold,
        dry_run=args.dry_run,
    )
