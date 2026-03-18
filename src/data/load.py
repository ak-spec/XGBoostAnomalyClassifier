import pandas as pd

def load_interactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df.columns = ["user", "item", "rating"]

    # enforce types
    df["user"] = df["user"].astype(int)
    df["item"] = df["item"].astype(int)
    df["rating"] = df["rating"].astype(float)

    return df

