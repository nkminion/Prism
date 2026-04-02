from .base import BaseScraper
from .coco import COCOScraper
from .inaturalist import INaturalistScraper
from .open_images import OpenImagesScraper
from .places365 import Places365Scraper
from .wikimedia import WikiMediaScraper

__all__ = [
    "BaseScraper",
    "OpenImagesScraper",
    "COCOScraper",
    "INaturalistScraper",
    "Places365Scraper",
    "WikiMediaScraper",
]
