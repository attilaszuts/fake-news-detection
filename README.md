# Fake news detection

This repository contains code for my pet project where I am using datasets from kaggle to build a classifier for detecting fake news.

## Data source

Use the [`kaggle`](https://github.com/Kaggle/kaggle-api) python package to download the datasets.

Datasets used:
* [fakenewsdataset](https://www.kaggle.com/datasets/sumanthvrao/fakenewsdataset)
* [fake-news competition](https://www.kaggle.com/competitions/fake-news/data)
* [fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Not used in V1)

## Data cleaning steps

Different datasets require different steps to clean them. First and foremost, you need to make sure that all your data is in the same shape, so that after this initial pre-processing we can create a single dataframe containing the training corpus and another one for testing.

### Fake-and-real-news-dataset

1. Create a single dataframe from `Fake.csv` and `True.csv`
2. Create a train-test split. (80-20)

### fake-news

There are no further steps necessary as the data is already split in a train-test fashion.

## Feature engineering

Since this is a NLP task feature engineering is about making sure our corpus is cleaned and well formatted to be used in our classifiers or neural networks.