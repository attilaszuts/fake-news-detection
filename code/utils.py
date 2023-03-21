from pathlib import Path

DATASETS = [
    "sumanthvrao/fakenewsdataset",
    "clmentbisaillon/fake-and-real-news-dataset"
]
COMPETITION = "fake-news"

ROOT_DIR = Path(__file__).parent.parent
EXTRACTED_SOURCE = f"{ROOT_DIR}/data/extracted"
RAW_PATH = f"{ROOT_DIR}/data/raw"
