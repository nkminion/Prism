import concurrent.futures
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import cv2
import imagehash
import numpy as np
from tqdm import tqdm

# Constants
_THUMB_SIZE = (256, 256)
_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# for multiprocessing
def _hash_image(args: tuple):
    img_path, hash_size = args
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return img_path, None

        img = cv2.resize(img, (hash_size + 1, hash_size))
        diff = img[:, 1:] > img[:, :-1]  # dhash

        return img_path, diff.flatten()
    except Exception:
        return img_path, None


def scan_dir(dirpath):
    images = []
    try:
        with os.scandir(dirpath) as it:
            subdirs, files = [], []
            for entry in it:
                if entry.name.startswith("."):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    subdirs.append(entry.path)
                elif os.path.splitext(entry.name)[1].lower() in _VALID_EXTENSIONS:
                    images.append(Path(entry.path))
    except PermissionError:
        return [], []
    return images, subdirs


def find_images(root, workers=8):
    all_images = []
    queue = [str(root)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        while queue:
            results = list(pool.map(scan_dir, queue))
            queue = []
            for images, subdirs in results:
                all_images.extend(images)
                queue.extend(subdirs)
    return all_images


class CrossSourceDeduplicator:
    """
    Remove near-duplicate images across sources.

    Uses perceptual hashing (pHash) - robust to:
    - Different resolutions
    - JPEG compression differences
    - Minor crops
    """

    def __init__(
        self, dataset_dir: Path, hash_size: int = 12, cache_file: Optional[Path] = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.hash_size = hash_size
        self.cache_file = (
            Path(cache_file) if cache_file else self.dataset_dir / ".hash_cache.pkl"
        )

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
        all_images = find_images(self.dataset_dir)

        print(f"  Found {len(all_images):,} images\n")

        if not all_images:
            return 0

        # Step 2: Load Cache
        cache = self._load_cache()

        need_hashing = [
            p for p in all_images if str(p) not in cache or cache[str(p)] is None
        ]

        print(f"  Cache hits : {len(all_images) - len(need_hashing):,}")
        print(f"  To hash    : {len(need_hashing):,}\n")

        # Step 3: Parallel Hashing
        if need_hashing:
            errors = 0
            args_list = [(p, self.hash_size) for p in need_hashing]

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Map the hashing function over all images
                futures = {
                    executor.submit(_hash_image, arg): arg[0] for arg in args_list
                }

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(need_hashing),
                    desc="  Hashing",
                ):
                    path, hash_array = future.result()
                    cache[str(path)] = hash_array
                    if hash_array is None:
                        errors += 1

            print(f"\n  Finished hashing (errors: {errors})")
            self._save_cache(cache)

        # Step 4: Filter valid entries and prepare for NumPy
        valid_paths = []
        hash_list = []

        # Only process files that still exist on disk and hashed successfully
        for path in tqdm(all_images, desc="  Preparing Matrix"):
            path_str = str(path)
            hash_arr = cache.get(path_str)
            if hash_arr is not None and path.exists():
                valid_paths.append(path)
                hash_list.append(hash_arr)

        print(f"  Valid hashes for comparison: {len(valid_paths):,}\n")

        if len(valid_paths) < 2:
            return 0

        # Step 5: Vectorized NumPy Comparison
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

        # Step 6: Remove or report
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

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self, cache_dict: dict) -> None:
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

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
        "--hash-size",
        "-s",
        type=int,
        default=8,
        help="Hash size (8=64-bit, 12=144-bit)",
    )
    parser.add_argument(
        "--cache", "-c", type=Path, default=None, help="Path to hash cache file"
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

    dedup = CrossSourceDeduplicator(
        dataset_dir=args.directory, hash_size=args.hash_size, cache_file=args.cache
    )

    dedup.deduplicate(
        threshold=args.threshold, dry_run=args.dry_run, max_workers=args.workers
    )
