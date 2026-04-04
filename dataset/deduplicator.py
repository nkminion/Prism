import concurrent.futures
import json
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm


# for multiprocessing
def _hash_image(img_path: Path, hash_size: int):
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            h = imagehash.phash(img, hash_size=hash_size)
            return img_path, h.hash.flatten()
    except Exception:
        return img_path, None


class CrossSourceDeduplicator:
    """
    Remove near-duplicate images across sources.

    Uses perceptual hashing (pHash) - robust to:
    - Different resolutions
    - JPEG compression differences
    - Minor crops
    """

    def __init__(self, dataset_dir: Path, hash_size: int = 12):
        self.dataset_dir = Path(dataset_dir)
        self.hash_size = hash_size

    def deduplicate(
        self,
        threshold: int = 8,
        dry_run: bool = False,
        max_workers: Optional[int] = None,
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
            max_workers: Max CPU cores to use for hashing. Defaults to all.

        Returns:
            Number of duplicates found/removed.
        """
        print(f"\n{'═' * 55}")
        print(f"  🔍 CROSS-SOURCE DEDUPLICATION")
        print(f"  Hash size: {self.hash_size} │ Threshold: {threshold}")
        print(f"{'═' * 55}\n")

        # Step 1: Collect all images
        extensions = {".jpg", ".jpeg", ".png"}
        all_images = [
            p
            for p in self.dataset_dir.rglob("*")
            if p.suffix.lower() in extensions and not p.name.startswith(".")
        ]
        print(f"  Found {len(all_images):,} images\n")

        if not all_images:
            return 0

        # Step 2: Parallel Hashing
        valid_paths = []
        hash_list = []
        errors = 0

        # Use ProcessPoolExecutor to max out CPU cores
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Map the hashing function over all images
            futures = {
                executor.submit(_hash_image, p, self.hash_size): p for p in all_images
            }

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(all_images),
                desc="  Hashing",
            ):
                path, hash_array = future.result()
                if hash_array is not None:
                    valid_paths.append(path)
                    hash_list.append(hash_array)
                else:
                    errors += 1

        print(f"\n  Hashed: {len(valid_paths):,} (errors: {errors})")

        if not valid_paths:
            return 0

        # Step 3: Vectorized NumPy Comparison
        # Stack all 1D boolean arrays into a single 2D matrix (N x Hash_Length)
        hash_matrix = np.vstack(hash_list)

        duplicates = []
        seen = set()

        for i in tqdm(range(len(valid_paths)), desc="  Comparing Matrix"):
            if valid_paths[i] in seen:
                continue

            # Vectorized XOR across the remaining rows, then sum to get Hamming distance
            dists = np.count_nonzero(hash_matrix[i] != hash_matrix[i + 1 :], axis=1)

            # Find indices where distance is within threshold
            match_indices = np.where(dists <= threshold)[0] + i + 1

            for j in match_indices:
                if valid_paths[j] not in seen:
                    keeper, removal = self._pick_keeper(valid_paths[i], valid_paths[j])

                    dist = int(dists[j - (i + 1)])
                    duplicates.append(
                        {
                            "removed": str(removal),
                            "kept": str(keeper),
                            "distance": dist,
                        }
                    )
                    seen.add(removal)

        print(f"\n  Found {len(duplicates):,} duplicates")

        # Step 4: Remove or report
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
                print("  (Dry run - no files deleted)")
                for d in duplicates[:5]:
                    print(
                        f"    Would remove: "
                        f"{Path(d['removed']).name} "
                        f"(keep: {Path(d['kept']).name}, "
                        f"dist={d['distance']})"
                    )

        return len(duplicates)

    def _pick_keeper(self, path1: Path, path2: Path) -> tuple:
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

        if _score(path1) >= _score(path2):
            return path1, path2
        return path2, path1


if __name__ == "__main__":
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description="Cross-source image deduplication")
    parser.add_argument("directory", type=Path, help="Directory to deduplicate")
    parser.add_argument(
        "--threshold", "-t", type=int, default=8, help="Hamming distance threshold"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Report only, don't delete"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Max CPU cores to use",
    )

    args = parser.parse_args()

    dedup = CrossSourceDeduplicator(args.directory)
    dedup.deduplicate(
        threshold=args.threshold, dry_run=args.dry_run, max_workers=args.workers
    )
