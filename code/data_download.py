import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

from utils import ROOT_DIR, COMPETITION, EXTRACTED_SOURCE

# Define competition, dataset and file names
compressed_source = f"{ROOT_DIR}/data/compressed"
files = [
    "fake-and-real-news-dataset.zip",
    "fake-news.zip",
    "fakenewsdataset.zip"
]


# Download data from Kaggle
api = KaggleApi()
api.authenticate()

for d in datasets:
    api.dataset_download_files(dataset=d, path=compressed_source)
api.competition_download_files(competition=COMPETITION, path=compressed_source)

# Extract data from zipfiles
for f in files:
    with zipfile.ZipFile(f"{compressed_source}/{f}", 'r') as zip_ref:
        zip_ref.extractall(f"{EXTRACTED_SOURCE}/{f}")
