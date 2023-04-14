from code.data import clean, save

if __name__ == "__main__":
    df = clean()
    save(df)