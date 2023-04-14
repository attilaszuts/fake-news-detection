import os
import re
import string
import zipfile
import nltk

import pandas as pd
from kaggle import KaggleApi
from nltk import WordNetLemmatizer
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from code.utils import RAW_PATH, CLEANED_PATH, ROOT_DIR, DATASETS, COMPETITION, EXTRACTED_SOURCE

# TODO: raise exception if no ~/.kaggle/kaggle.json
compressed_source = f"{ROOT_DIR}/data/compressed"
files = [
    "fake-and-real-news-dataset.zip",
    "fake-news.zip",
    "fakenewsdataset.zip"
]


def download() -> None:
    # Download data from Kaggle
    api = KaggleApi()
    api.authenticate()

    for dataset in DATASETS:
        api.dataset_download_files(dataset=dataset, path=compressed_source)
    api.competition_download_files(competition=COMPETITION, path=compressed_source)

    # Extract data from zipfiles
    for f in files:
        with zipfile.ZipFile(f"{compressed_source}/{f}", 'r') as zip_ref:
            zip_ref.extractall(f"{EXTRACTED_SOURCE}/{f}")


def clean() -> pd.DataFrame:
    df = pd.read_csv(f"{RAW_PATH}/train.csv")
    i = nltk.corpus.stopwords.words('english')
    j = list(string.punctuation)

    stopwords = set(i).union(j)
    wordnet_lemma = WordNetLemmatizer()

    def preprocess(row):
        new_row = []
        row = re.sub('[^a-z\s]', '', row.lower())  # get rid of noise
        row = [w for w in row.split() if w not in set(stopwords)]  # remove stopwords
        for word in row:
            new_word = wordnet_lemma.lemmatize(word)
            new_row.append(new_word)
        return " ".join(new_row)

    df = df[pd.isnull(df.text) == False]  # drop rows with no text
    df["cleaned_text"] = df["text"].apply(preprocess)
    df = df[["cleaned_text", "label"]]
    return df


def save(df: pd.DataFrame) -> None:
    if not os.path.exists(CLEANED_PATH):
        os.mkdir(f"{CLEANED_PATH}")
    df.to_csv(f"{CLEANED_PATH}/train.csv")


def process_fake_news() -> tuple[DataFrame, DataFrame]:
    """
    Read in train_df and test_df data.

    """
    train_df = pd.read_csv(f"{EXTRACTED_SOURCE}/fake-news.zip/train.csv")
    test_df = pd.read_csv(f"{EXTRACTED_SOURCE}/fake-news.zip/test.csv")
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    return train_df, test_df


def process_fake_and_real_news():
    true = pd.read_csv(f"{EXTRACTED_SOURCE}/fake-and-real-news-dataset.zip/true.csv")
    fake = pd.read_csv(f"{EXTRACTED_SOURCE}/fake-and-real-news-dataset.zip/fake.csv")

    true.label = 0
    fake.label = 1

    merged = pd.concat([true, fake])

    train_df, test_df = train_test_split(merged, test_size=0.2, random_state=20230313)
    return train_df, test_df


def combine_datasets(train_dfs: list[DataFrame], test_dfs: list[DataFrame]) -> tuple[DataFrame, DataFrame]:

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    return train_df, test_df


def create_raw() -> None:
    train1, test1 = process_fake_news()
    train2, test2 = process_fake_and_real_news()

    train, test = combine_datasets([train1, train2], [test1, test2])

    if not os.path.exists(RAW_PATH):
        os.mkdir(f"{RAW_PATH}")
    train.to_csv(f"{RAW_PATH}/train.csv", index=False)
    test.to_csv(f"{RAW_PATH}/test.csv", index=False)