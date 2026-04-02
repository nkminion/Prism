import csv
import time
from pathlib import Path

import config
from tqdm import tqdm

from .base import BaseScraper


class WikiMediaScraper(BaseScraper):
    """
    WikiMedia Commons - CC-BY-SA / CC0 / Public Domain.

    All content on Commons has explicit licensing.
    We only download CC0 / CC-BY / CC-BY-SA / Public Domain.

    Rate limit: Be polite - 1 request/second :).
    User-Agent with contact info required.

    https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia
    """

    API_BASE = "https://commons.wikimedia.org/w/api.php"
    ACCESS_TOKEN_URL = "https://meta.wikimedia.org/w/rest.php/oauth2/access_token"

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, "wikimedia")
        self.access_token = config.WIKIMEDIA_ACCESS_TOKEN
        self.session.headers.update(
            {
                # "Authorization": f"Bearer {self.access_token}",
                "User-Agent": (
                    "PrismDatasetBot/1.0 "
                    "(open-source ML dataset; "
                    "contact: code@pythonicvarun.me) "
                    "python-requests"
                ),
            }
        )

    def _get_category_images(
        self,
        category_name: str,
        limit: int = 500,
    ) -> list:
        """
        Get image file info from a Commons category.
        Recursively fetches paginated results.
        """
        images = []
        gcm_continue = None

        while len(images) < limit:
            time.sleep(1.0)  # Polite rate limit

            params = {
                "action": "query",
                "generator": "categorymembers",
                "gcmtitle": f"Category:{category_name}",
                "gcmtype": "file",
                "gcmlimit": min(50, limit - len(images)),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": 1024,  # Get thumbnail URL too
                "format": "json",
            }

            if gcm_continue:
                params["gcmcontinue"] = gcm_continue

            try:
                resp = self.session.get(
                    self.API_BASE,
                    params=params,
                    timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()
                # print(f"Data: {data}")
            except Exception as e:
                self.logger.warning(f"WikiMedia error: {e}")
                break

            pages = data.get("query", {}).get("pages", {})

            for page_id, page in pages.items():
                info_list = page.get("imageinfo", [])
                if not info_list:
                    continue
                info = info_list[0]

                # Check mime type
                mime = info.get("mime", "")
                if mime not in ("image/jpeg", "image/png"):
                    continue

                # Check size
                w = info.get("width", 0)
                h = info.get("height", 0)
                if w < 256 or h < 256:
                    continue

                # Extract license from metadata
                ext = info.get("extmetadata", {})
                license_short = (
                    ext.get("LicenseShortName", {}).get("value", "unknown").lower()
                )

                # Only allow open licenses
                is_allowed = any(
                    allowed in license_short
                    for allowed in [
                        "cc0",
                        "cc-by",
                        "cc by",
                        "public domain",
                        "pd",
                    ]
                )
                # Block NC and ND
                is_blocked = any(blocked in license_short for blocked in ["nc", "nd"])

                if not is_allowed or is_blocked:
                    continue

                url = info.get("url", "")
                # Use thumbnail if original is too large
                thumb_url = info.get("thumburl", url)

                artist = ext.get("Artist", {}).get("value", "unknown")

                images.append(
                    {
                        "id": str(page_id),
                        "url": thumb_url if thumb_url else url,
                        "url_full": url,
                        "title": page.get("title", ""),
                        "license": license_short,
                        "artist": artist,
                    }
                )

            # Pagination
            cont = data.get("continue", {})
            gcm_continue = cont.get("gcmcontinue")
            if not gcm_continue:
                break

        return images[:limit]

    def scrape(
        self,
        category: str,
        wiki_categories: list,
        target_count: int,
    ) -> int:
        """
        Download CC-licensed images from WikiMedia Commons.
        """
        if not wiki_categories:
            return 0

        self.logger.info(
            f"WikiMedia: {category} " f"(cats={wiki_categories}, target={target_count})"
        )

        save_dir = self.output_dir / category / self.source_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Attribution file
        attr_file = save_dir / "ATTRIBUTION.csv"
        attr_exists = attr_file.exists() and attr_file.stat().st_size > 0
        attr_handle = open(attr_file, "a", newline="")
        attr_writer = csv.writer(attr_handle)
        if not attr_exists:
            attr_writer.writerow(
                [
                    "filename",
                    "page_id",
                    "title",
                    "artist",
                    "license",
                    "source_url",
                ]
            )

        existing = self.get_existing_count(category)
        remaining = target_count - existing
        if remaining <= 0:
            attr_handle.close()
            return existing

        per_cat = remaining // len(wiki_categories) + 1
        downloaded = 0

        for wiki_cat in wiki_categories:
            if downloaded >= remaining:
                break

            self.logger.info(f"  Category:{wiki_cat} " f"(target: {per_cat})")

            images = self._get_category_images(
                wiki_cat,
                limit=per_cat,
            )
            self.logger.info(f"    Found {len(images)} valid CC images")

            for img in tqdm(
                images,
                desc=f"    {wiki_cat[:25]}",
                leave=False,
            ):
                if downloaded >= remaining:
                    break

                if img["id"] in self.downloaded_ids:
                    continue

                save_path = save_dir / f"wiki_{img['id']}.jpg"

                if self.download_image(img["url"], save_path):
                    self._mark_downloaded(img["id"])
                    downloaded += 1

                    # Attribution
                    attr_writer.writerow(
                        [
                            save_path.name,
                            img["id"],
                            img["title"],
                            img["artist"][:100],
                            img["license"],
                            img["url_full"],
                        ]
                    )

        attr_handle.close()
        total = existing + downloaded
        self.logger.info(
            f"✅ WikiMedia {category}: " f"{downloaded} new, {total} total"
        )
        return total
