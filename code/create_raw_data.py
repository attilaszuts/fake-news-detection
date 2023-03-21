import os

import pandas as pd
from pandas import DataFrame

from sklearn.model_selection import train_test_split

from utils import RAW_PATH, EXTRACTED_SOURCE


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

    merged = true.concat(fake)

    train_df, test_df = train_test_split(merged, test_size=0.2, random_state=20230313)
    return train_df, test_df


def combine_datasets(train_dfs: list[DataFrame], test_dfs: list[DataFrame]) -> tuple[DataFrame, DataFrame]:
    train_df = train_dfs[0]
    for d in train_dfs[1:]:
        train_df.concat(d)

    test_df = test_dfs[0]
    for d in test_dfs[1:]:
        test_df.concat(d)

    return train_df, test_df


if __name__ == "__main__":
    train1, test1 = process_fake_news()
    train2, test2 = process_fake_and_real_news()

    train, test = combine_datasets([train1, train2], [test1, test2])

    if not os.path.exists(RAW_PATH):
        os.mkdir(f"{RAW_PATH}")
    train.to_csv(f"{RAW_PATH}/train.csv")
    test.to_csv(f"{RAW_PATH}/test.csv")
