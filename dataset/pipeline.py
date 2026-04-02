import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from config import CATEGORIES, RAW_DIR, SOURCE_ALLOCATION
from scrapers import (
    COCOScraper,
    INaturalistScraper,
    OpenImagesScraper,
    Places365Scraper,
    WikiMediaScraper,
)


class ScrapingPipeline:
    """
    Execution order: bulk datasets first then APIs (rate-limited).
    """

    SOURCE_ORDER = [
        "coco",
        "open_images",
        "places365",
        "inaturalist",
        "wikimedia",
    ]

    def __init__(self):
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        self.scrapers = {
            "coco": COCOScraper(RAW_DIR),
            "open_images": OpenImagesScraper(RAW_DIR),
            "places365": Places365Scraper(RAW_DIR),
            "inaturalist": INaturalistScraper(RAW_DIR),
            "wikimedia": WikiMediaScraper(RAW_DIR),
        }
        self.stats = {}
        self.errors = []

    def _run_source_for_category(
        self,
        source: str,
        category: str,
        target: int,
    ) -> int:
        """Dispatch to the correct scraper."""
        cfg = CATEGORIES[category]
        scraper = self.scrapers[source]

        try:
            if source == "coco":
                if not cfg.coco_classes:
                    return 0
                return scraper.scrape(
                    category,
                    cfg.coco_classes,
                    target,
                )

            elif source == "open_images":
                if not cfg.open_images_labels:
                    return 0
                return scraper.scrape(
                    category,
                    cfg.open_images_labels,
                    target,
                )

            elif source == "places365":
                if not cfg.places365_categories:
                    return 0
                return scraper.scrape(
                    category,
                    cfg.places365_categories,
                    target,
                )

            elif source == "inaturalist":
                if not cfg.inaturalist_taxon_ids:
                    return 0
                return scraper.scrape(
                    category,
                    cfg.inaturalist_taxon_ids,
                    target,
                )

            elif source == "wikimedia":
                if not cfg.wikimedia_categories:
                    return 0
                return scraper.scrape(
                    category,
                    cfg.wikimedia_categories,
                    target,
                )

            return 0

        except KeyboardInterrupt:
            raise
        except Exception as e:
            err = f"{source}/{category}: {str(e)[:200]}"
            self.errors.append(err)
            print(f"    ❌ ERROR: {err}")
            return 0

    def run(self, categories=None, sources=None):
        """
        Run full scraping pipeline.

        Args:
            categories: List of category names, or None for all.
            sources:    List of source names, or None for all.
        """
        cats = categories or list(CATEGORIES.keys())
        srcs = sources or self.SOURCE_ORDER

        start = time.time()

        # Process by SOURCE for better efficiency
        for source in srcs:
            phase_start = time.time()

            print(f"\n{'━' * 60}")
            print(f"🔧 SOURCE: {source.upper()}")
            print(f"{'━' * 60}")

            for cat_name in cats:
                alloc = SOURCE_ALLOCATION.get(cat_name, {})
                frac = alloc.get(source, 0)
                if frac <= 0:
                    continue

                target = int(CATEGORIES[cat_name].target_count * frac)

                print(f"\n  📁 {cat_name} ← {source} " f"(target: {target:,})")

                count = self._run_source_for_category(
                    source,
                    cat_name,
                    target,
                )

                self.stats[f"{cat_name}/{source}"] = count
                print(f"     Result: {count:,} / {target:,}")

            phase_elapsed = time.time() - phase_start
            print(f"\n  ⏱️  {source} phase: " f"{phase_elapsed / 60:.1f} min")

        elapsed = time.time() - start
        self._print_report(elapsed, cats)
        self._save_report(elapsed)

    def _print_report(self, elapsed: float, cats: list):
        """LLM Generated: Print final summary report."""
        print(f"\n{'═' * 62}")
        print(f"  📊 FINAL REPORT")
        print(f"  Time: {timedelta(seconds=int(elapsed))}")
        print(f"{'═' * 62}\n")

        grand_total = 0

        for cat in cats:
            target = CATEGORIES[cat].target_count
            cat_total = sum(self.stats.get(f"{cat}/{s}", 0) for s in self.SOURCE_ORDER)
            grand_total += cat_total

            pct = cat_total / target * 100 if target else 0
            filled = min(int(pct / 5), 20)
            bar = "█" * filled + "░" * (20 - filled)

            icon = "✅" if pct >= 90 else "🟡" if pct >= 60 else "❌"

            print(
                f"  {icon} {cat:12s} {bar} "
                f"{cat_total:>6,} / {target:>6,} "
                f"({pct:.0f}%)"
            )

            # Per-source breakdown
            alloc = SOURCE_ALLOCATION.get(cat, {})
            for src in self.SOURCE_ORDER:
                if src not in alloc:
                    continue
                count = self.stats.get(f"{cat}/{src}", 0)
                src_target = int(target * alloc[src])
                print(f"       {src:15s}: " f"{count:>5,} / {src_target:>5,}")

        print(f"\n  {'─' * 48}")
        print(f"  GRAND TOTAL: {grand_total:>7,} / 100,000")

        if self.errors:
            print(f"\n  ⚠️  {len(self.errors)} errors:")
            for err in self.errors[:10]:
                print(f"     • {err}")

        print(f"{'═' * 62}")

    def _save_report(self, elapsed: float):
        """Save report to JSON."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_hours": round(elapsed / 3600, 2),
            "stats": self.stats,
            "errors": self.errors,
            "grand_total": sum(self.stats.values()),
            "sources_used": list({k.split("/")[1] for k in self.stats}),
            "license_clean": True,
        }
        report_path = RAW_DIR / "scraping_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  💾 Report: {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Colorization Dataset Pipeline " "(License-Clean)"
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="*",
        default=None,
        choices=list(CATEGORIES.keys()),
        help="Categories to process (default: all)",
    )
    parser.add_argument(
        "--sources",
        "-s",
        nargs="*",
        default=None,
        choices=[
            "coco",
            "open_images",
            "places365",
            "inaturalist",
            "wikimedia",
        ],
        help="Sources to use (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without downloading",
    )

    args = parser.parse_args()

    if args.dry_run:
        from config import verify_config

        verify_config()
    else:
        pipeline = ScrapingPipeline()
        pipeline.run(
            categories=args.categories,
            sources=args.sources,
        )
