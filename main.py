from code.data import clean, save, download, create_raw

if __name__ == "__main__":
    print("downloading...")
    download()
    print("creating raw...")
    create_raw()
    print("cleaning...")
    df = clean()
    print("saving...")
    save(df)
    print("done.")
