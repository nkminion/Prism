import csv
import time
from pathlib import Path

import config
from tqdm import tqdm

from .base import BaseScraper


class INaturalistScraper(BaseScraper):
    """
    iNaturalist - CC-licensed research-grade observations.

    ONLY downloads photos with explicit CC0/CC-BY/CC-BY-SA licenses.
    Rate limit: 60 requests/minute (be respectful).

    License info: https://www.inaturalist.org/pages/terms
    Photos are owned by observers - we filter for open CC licenses.
    """

    API_BASE = "https://api.inaturalist.org/v1"

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, "inaturalist")
        self.session.headers.update(
            {
                "User-Agent": (
                    "PrismDatasetBot/1.0 "
                    "(open-source image colorization research dataset; "
                    "contact: code@pythonicvarun.me)"
                )
            }
        )
        self._request_times = []

    # Rate Limiting
    def _rate_limit(self):
        """Enforce 60 requests/minute with safety margin."""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= 55:
            sleep_time = 60 - (now - self._request_times[0]) + 2
            if sleep_time > 0:
                self.logger.debug(f"Rate limit sleep {sleep_time:.0f}s")
                time.sleep(sleep_time)
        self._request_times.append(time.time())

    # API
    def _fetch_observations(
        self,
        taxon_id: int,
        page: int = 1,
        per_page: int = 200,
    ) -> list:
        """
        Fetch research-grade observations with CC-licensed photos.

        Key params:
        - quality_grade=research     - verified identifications
        - photo_licensed=true        - has a license
        - license=cc0,cc-by,cc-by-sa - ONLY open licenses
        """
        self._rate_limit()

        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "quality_grade": "research",
            "photo_licensed": "true",
            "license": "cc0,cc-by,cc-by-sa",
            "per_page": per_page,
            "page": page,
            "order": "desc",
            "order_by": "votes",
        }

        try:
            resp = self.session.get(
                f"{self.API_BASE}/observations",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            self.logger.warning(f"iNat API error: {e}")
            return []

    def _extract_photos(self, observations: list) -> list:
        """Extract photo URLs + attribution from observations."""
        photos = []
        for obs in observations:
            observer = obs.get("user", {}).get("login", "unknown")
            obs_id = obs.get("id", "")

            for photo in obs.get("photos", []):
                url = photo.get("url", "")
                if not url:
                    continue

                photo_id = str(photo.get("id", ""))
                license_code = photo.get("license_code", "")

                # Double-check license
                if license_code not in config.INATURALIST_ALLOWED_LICENSES:
                    continue

                # Get larger version (replace square with large)
                large_url = url.replace("square", "large")
                medium_url = url.replace("square", "medium")

                if photo_id not in self.downloaded_ids:
                    photos.append(
                        {
                            "id": photo_id,
                            "url_large": large_url,
                            "url_medium": medium_url,
                            "license": license_code,
                            "observer": observer,
                            "obs_id": obs_id,
                            "obs_url": (
                                f"https://www.inaturalist.org/" f"observations/{obs_id}"
                            ),
                        }
                    )
        return photos

    def scrape(
        self,
        category: str,
        taxon_ids: list,
        target_count: int,
    ) -> int:
        """
        Download CC-licensed photos for given taxonomic groups.

        Args:
            category:     e.g., "animals"
            taxon_ids:    iNat taxon IDs, e.g., [40151] for Mammalia
            target_count: images to download
        """
        if not taxon_ids:
            return 0

        self.logger.info(
            f"iNaturalist: {category} " f"(taxa={taxon_ids}, target={target_count})"
        )

        save_dir = self.output_dir / category / self.source_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Attribution file (CC license requirement)
        attr_file = save_dir / "ATTRIBUTION.csv"
        attr_exists = attr_file.exists() and attr_file.stat().st_size > 0
        attr_handle = open(attr_file, "a", newline="")
        attr_writer = csv.writer(attr_handle)
        if not attr_exists:
            attr_writer.writerow(
                [
                    "filename",
                    "photo_id",
                    "observer",
                    "license",
                    "observation_url",
                ]
            )

        existing = self.get_existing_count(category)
        remaining = target_count - existing
        if remaining <= 0:
            attr_handle.close()
            return existing

        per_taxon = remaining // len(taxon_ids) + 1
        downloaded = 0

        for taxon_id in taxon_ids:
            self.logger.info(f"  Taxon {taxon_id}: targeting {per_taxon}")
            page = 1
            taxon_dl = 0
            max_pages = 40  # 40 * 200 = 8K max per taxon

            pbar = tqdm(
                total=per_taxon,
                desc=f"    Taxon {taxon_id}",
                leave=False,
            )

            while taxon_dl < per_taxon and page <= max_pages:
                observations = self._fetch_observations(
                    taxon_id,
                    page=page,
                )
                if not observations:
                    break

                photos = self._extract_photos(observations)

                for photo in photos:
                    if taxon_dl >= per_taxon:
                        break

                    save_path = save_dir / f"inat_{taxon_id}_{photo['id']}.jpg"

                    # Try large first, then medium
                    ok = self.download_image(
                        photo["url_large"],
                        save_path,
                    )
                    if not ok:
                        ok = self.download_image(
                            photo["url_medium"],
                            save_path,
                        )

                    if ok:
                        self._mark_downloaded(photo["id"])
                        taxon_dl += 1
                        downloaded += 1
                        pbar.update(1)

                        # Record attribution
                        attr_writer.writerow(
                            [
                                save_path.name,
                                photo["id"],
                                photo["observer"],
                                photo["license"],
                                photo["obs_url"],
                            ]
                        )

                page += 1
                time.sleep(0.3)

            pbar.close()

        attr_handle.close()
        total = existing + downloaded
        self.logger.info(
            f"✅ iNaturalist {category}: "
            f"{downloaded} new (all CC-licensed), {total} total"
        )
        return total
