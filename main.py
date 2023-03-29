from code.create_raw_data import create_raw
from code.data_cleaning import clean
from code.data_download import download

if __name__ == "__main__":
    download()
    clean()
    create_raw()
