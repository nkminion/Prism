import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

# API KEYS
# WikiMedia: https://www.mediawiki.org/wiki/OAuth/For_Developers
WIKIMEDIA_ACCESS_TOKEN = os.getenv("WIKIMEDIA_ACCESS_TOKEN")

# Paths
RAW_DIR = Path("raw_downloads")
OUTPUT_DIR = Path("colorization_dataset_v1")

# Cache dirs for bulk datasets (downloaded once)
COCO_CACHE = RAW_DIR / ".coco_cache"
OI_CACHE = RAW_DIR / ".open_images_cache"
PLACES_CACHE = RAW_DIR / ".places365_cache"


# Accepted licenses
INATURALIST_ALLOWED_LICENSES = {
    "cc0",  # Public domain
    "cc-by",  # Attribution
    "cc-by-sa",  # Attribution-ShareAlike
}

WIKIMEDIA_ALLOWED_LICENSES = {
    "cc0",
    "cc-by",
    "cc-by-sa",
    "cc-by-2.0",
    "cc-by-sa-2.0",
    "cc-by-3.0",
    "cc-by-sa-3.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "public domain",
}


@dataclass
class CategoryConfig:
    name: str
    target_count: int
    open_images_labels: List[str] = field(default_factory=list)
    coco_classes: List[str] = field(default_factory=list)
    places365_categories: List[str] = field(default_factory=list)
    inaturalist_taxon_ids: List[int] = field(default_factory=list)
    wikimedia_categories: List[str] = field(default_factory=list)


CATEGORIES: Dict[str, CategoryConfig] = {
    # 1. Humans / Portraits (15K)
    "humans": CategoryConfig(
        name="humans",
        target_count=15_000,
        open_images_labels=[
            "/m/01g317",  # Person
            "/m/0dzct",  # Human face
            "/m/04yx4",  # Man
            "/m/03bt1vf",  # Woman
            "/m/01bl7v",  # Boy
            "/m/05s2s",  # Girl
        ],
        coco_classes=[
            "person",
        ],
        places365_categories=[],
        inaturalist_taxon_ids=[],
        wikimedia_categories=[
            "Portrait_photographs",
            "People_by_activity",
        ],
    ),
    # 2. Animals / Wildlife (12K)
    "animals": CategoryConfig(
        name="animals",
        target_count=12_000,
        open_images_labels=[
            "/m/0bt9lr",  # Dog
            "/m/01yrx",  # Cat
            "/m/03k3r",  # Horse
            "/m/09686",  # Butterfly
            "/m/0ch_cf",  # Lion
            "/m/01280g",  # Bird
            "/m/0jbk",  # Fish
            "/m/03qrc",  # Reptile
            "/m/068hy",  # Parrot
        ],
        coco_classes=[
            "dog",
            "cat",
            "horse",
            "bird",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "sheep",
            "cow",
        ],
        inaturalist_taxon_ids=[
            40151,  # Mammalia
            3,  # Aves (Birds)
            26036,  # Reptilia
            47120,  # Insecta
            47178,  # Ray-finned fish
            20978,  # Amphibia
        ],
        wikimedia_categories=[
            "Wildlife_photographs",
            "Photographs_of_dogs",
            "Photographs_of_cats",
        ],
    ),
    # 3. Scenery / Landscapes (12K)
    "scenery": CategoryConfig(
        name="scenery",
        target_count=12_000,
        open_images_labels=[
            "/m/09d_r",  # Mountain
            "/m/0csby",  # Cloud
            "/m/05h0n",  # Nature
            "/m/09t49",  # Grassland
            "/m/01bqvp",  # Sky
        ],
        places365_categories=[
            "mountain",
            "mountain_path",
            "mountain_snowy",
            "field/cultivated",
            "field/wild",
            "forest_path",
            "forest/broadleaf",
            "canyon",
            "cliff",
            "coast",
            "coastline",
            "desert/sand",
            "desert/vegetation",
            "lake/natural",
            "valley",
            "volcano",
            "marsh",
            "glacier",
            "rainforest",
            "tundra",
        ],
        wikimedia_categories=[
            "Landscape_photographs",
            "Mountains",
            "Deserts",
            "Waterfalls",
            "Valleys",
        ],
    ),
    # 4. Vehicles / Urban (10K)
    "vehicles": CategoryConfig(
        name="vehicles",
        target_count=10_000,
        open_images_labels=[
            "/m/0k4j",  # Car
            "/m/07yv9",  # Vehicle
            "/m/01bjv",  # Bus
            "/m/07r04",  # Truck
            "/m/019jd",  # Boat
            "/m/0cmf2",  # Airplane
            "/m/04_sv",  # Motorcycle
            "/m/0199g",  # Bicycle
            "/m/07jdr",  # Train
        ],
        coco_classes=[
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "airplane",
            "boat",
            "train",
        ],
        wikimedia_categories=[
            "Automobiles",
            "Motorcycles",
            "Aircraft",
            "Ships",
            "Trains",
            "Bicycles",
        ],
    ),
    # 5. Food / Cuisine (10K)
    "food": CategoryConfig(
        name="food",
        target_count=10_000,
        open_images_labels=[
            "/m/02wbm",  # Food
            "/m/0270h",  # Dessert
            "/m/0fszt",  # Cake
            "/m/0l515",  # Sushi
            "/m/0fj52s",  # Salad
            "/m/01sh3",  # Bread
        ],
        coco_classes=[
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "pizza",
            "donut",
            "cake",
        ],
        wikimedia_categories=[
            "Food_photographs",
            "Desserts",
            "Fruit",
            "Dishes_(food)",
        ],
    ),
    # 6. Buildings / Architecture (10K)
    "buildings": CategoryConfig(
        name="buildings",
        target_count=10_000,
        open_images_labels=[
            "/m/0cgh4",  # Building
            "/m/02dgv",  # Castle
            "/m/03jm5",  # Tower
            "/m/0h8lhkg",  # Skyscraper
            "/m/03gq5hm",  # Church
            "/m/06_72j",  # Bridge
        ],
        places365_categories=[
            "tower",
            "skyscraper",
            "church/outdoor",
            "church/indoor",
            "mosque/outdoor",
            "palace",
            "bridge",
            "lighthouse",
            "temple/asia",
            "castle",
            "cathedral/outdoor",
            "cathedral/indoor",
        ],
        wikimedia_categories=[
            "Gothic_architecture",
            "Art_Nouveau_architecture",
            "Modern_architecture",
            "Castles_in_Europe",
            "Bridges",
            "Lighthouses",
            "Skyscrapers",
            "Mosques",
            "Buddhist_temples",
        ],
    ),
    # 7. Indoor Scenes (8K)
    "indoor": CategoryConfig(
        name="indoor",
        target_count=8_000,
        open_images_labels=[
            "/m/02crq1",  # Couch
            "/m/01mzpv",  # Chair
            "/m/04bcr3",  # Table
            "/m/079cl",  # Kitchen
            "/m/0d4v4",  # Window
        ],
        coco_classes=[
            "couch",
            "chair",
            "dining table",
            "bed",
            "tv",
            "laptop",
            "sink",
            "refrigerator",
            "oven",
            "microwave",
            "toilet",
        ],
        places365_categories=[
            "living_room",
            "kitchen",
            "bedroom",
            "bathroom",
            "dining_room",
            "library/indoor",
            "office",
            "home_office",
            "restaurant",
            "bar",
            "museum/indoor",
            "lobby",
            "classroom",
            "hotel_room",
        ],
        wikimedia_categories=[
            "Interior_design",
            "Kitchens",
            "Libraries",
        ],
    ),
    # 8. Flowers / Plants (8K)
    "flowers": CategoryConfig(
        name="flowers",
        target_count=8_000,
        open_images_labels=[
            "/m/0c9ph5",  # Flower
            "/m/07j7r",  # Tree
            "/m/0gqbt",  # Rose
        ],
        coco_classes=["potted plant"],
        inaturalist_taxon_ids=[
            47125,  # Magnoliopsida (flowering plants)
            47126,  # Plantae (kingdom)
        ],
        wikimedia_categories=[
            "Flowers",
            "Roses",
            "Orchids",
            "Tulips",
            "Sunflowers",
            "Garden_plants",
            "Cacti",
        ],
    ),
    # 9. Ocean / Water (8K)
    "ocean": CategoryConfig(
        name="ocean",
        target_count=8_000,
        open_images_labels=[
            "/m/06npx",  # Sea
            "/m/0b3yr",  # Beach
            "/m/04k94",  # Lake
            "/m/0dl1t",  # Waterfall
        ],
        places365_categories=[
            "ocean",
            "coast",
            "beach",
            "beach_house",
            "lake/natural",
            "waterfall",
            "river",
            "swimming_pool/outdoor",
            "harbor",
            "pier",
            "underwater/ocean_deep",
        ],
        wikimedia_categories=[
            "Ocean",
            "Beaches",
            "Lakes",
            "Rivers",
            "Coral_reefs",
            "Waves",
        ],
    ),
    # 10. Night / Low-Light (7K)
    "night": CategoryConfig(
        name="night",
        target_count=7_000,
        open_images_labels=[
            "/m/01d74z",  # Night
        ],
        places365_categories=[
            "downtown",
            "street",
            "highway",
            "gas_station",
            "amusement_park",
        ],
        wikimedia_categories=[
            "Night_photography",
            "Nighttime",
            "Fireworks",
            "Neon_signs",
            "City_lights",
            "Astrophotography",
            "Aurora_borealis",
            "Lightning",
        ],
    ),
}

SOURCE_ALLOCATION: Dict[str, Dict[str, float]] = {
    "humans": {
        "open_images": 0.50,  #  7,500
        "coco": 0.40,  #  6,000
        "wikimedia": 0.10,  #  1,500
    },
    "animals": {
        "open_images": 0.30,  #  3,600
        "coco": 0.20,  #  2,400
        "inaturalist": 0.40,  #  4,800
        "wikimedia": 0.10,  #  1,200
    },
    "scenery": {
        "open_images": 0.25,  #  3,000
        "places365": 0.55,  #  6,600
        "wikimedia": 0.20,  #  2,400
    },
    "vehicles": {
        "coco": 0.40,  #  4,000
        "open_images": 0.40,  #  4,000
        "wikimedia": 0.20,  #  2,000
    },
    "food": {
        "open_images": 0.40,  #  4,000
        "coco": 0.30,  #  3,000
        "wikimedia": 0.30,  #  3,000
    },
    "buildings": {
        "open_images": 0.30,  #  3,000
        "places365": 0.30,  #  3,000
        "wikimedia": 0.40,  #  4,000
    },
    "indoor": {
        "open_images": 0.20,  #  1,600
        "coco": 0.35,  #  2,800
        "places365": 0.35,  #  2,800
        "wikimedia": 0.10,  #    800
    },
    "flowers": {
        "open_images": 0.25,  #  2,000
        "coco": 0.05,  #    400
        "inaturalist": 0.45,  #  3,600
        "wikimedia": 0.25,  #  2,000
    },
    "ocean": {
        "open_images": 0.25,  #  2,000
        "places365": 0.45,  #  3,600
        "wikimedia": 0.30,  #  2,400
    },
    "night": {
        "open_images": 0.15,  #  1,050
        "places365": 0.40,  #  2,800
        "wikimedia": 0.45,  #  3,150
    },
}


def verify_config():
    """LLM Generated: Verify configuration integrity."""
    total = sum(c.target_count for c in CATEGORIES.values())

    print(f"\n{'═' * 62}")
    print(f"  📊 DATASET CONFIG")
    print(f"  ✅ COCO │ Open Images │ Places365 │ iNat │ WikiMedia")
    print(f"{'═' * 62}")
    print(f"  Categories: {len(CATEGORIES)}  │  " f"Total: {total:,} images\n")

    all_ok = True
    for name, cfg in CATEGORIES.items():
        alloc = SOURCE_ALLOCATION.get(name, {})
        alloc_sum = sum(alloc.values())
        ok = 0.99 <= alloc_sum <= 1.01
        if not ok:
            all_ok = False

        status = "✅" if ok else "❌"
        n_sources = len(alloc)

        print(
            f"  {status} {name:12s} │ "
            f"{cfg.target_count:>6,} imgs │ "
            f"{n_sources} sources │ "
            f"alloc={alloc_sum:.0%}"
        )

        for src, frac in alloc.items():
            count = int(cfg.target_count * frac)
            bar_len = int(frac * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"       {src:15s} {bar} {count:>5,}")
        print()

    print(f"  {'─' * 50}")
    print(f"  Grand Total: {total:>7,}")
    print(f"  Config valid: {'✅ YES' if all_ok else '❌ NO'}")
    print(f"{'═' * 62}")
    return all_ok


if __name__ == "__main__":
    verify_config()
