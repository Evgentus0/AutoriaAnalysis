import pymongo
import pandas as pd

DB_CONNECTION = "mongodb://localhost:27017"
DB_NAME = "Scrapper"
COLLECTION_NAME = "CarInfo"

PATH_TO_CSV = "data.csv"


def get_data_mongo() -> pd.DataFrame:
    client = pymongo.MongoClient(DB_CONNECTION)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    cursor = collection.find()
    df = pd.DataFrame(list(cursor))

    # Process data

    del df['_id']
    del df['NoviceType']

    df.loc[df["RaceKmThs"] == 0, "RaceKmThs"] = df["RaceKmThs"].mean()

    df["PriceUsd"] = df["PriceUsd"].apply(pd.to_numeric)
    df["PriceUah"] = df["PriceUah"].apply(pd.to_numeric)

    df["PriceUsd"] = df["PriceUsd"].replace(0, df["PriceUsd"].mean())
    df["PriceUah"] = df["PriceUah"].replace(0, df["PriceUah"].mean())

    return df


def get_data_csv() -> pd.DataFrame:
    df = pd.read_csv(PATH_TO_CSV)
    del df['ModelName']
    del df["AutoId"]
    del df["PriceUah"]
    return df
