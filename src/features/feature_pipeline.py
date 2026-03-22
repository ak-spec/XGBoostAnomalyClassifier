import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy

class BaselineFeatureBuilder:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # remove duplicate user-item interactions
        df = df.groupby(["user", "item"])["rating"].mean().reset_index()

        user_features = df.groupby("user").agg(
            num_unique_items_rated=("item", "nunique"),

            mean_rating=("rating", "mean"),
            median_rating=("rating", "median"),
            std_rating=("rating", "std"),
            min_rating=("rating", "min"),
            max_rating=('rating', 'max'),

            high_rating_ratio=("rating", lambda x: (x >= 4).mean()),
            low_rating_ratio=("rating", lambda x: (x <= 2).mean()),

            rating_0_pct=("rating", lambda x: (x == 0).mean()),
            rating_1_pct=("rating", lambda x: (x == 1).mean()),
            rating_4_pct=("rating", lambda x: (x == 4).mean()),
            rating_5_pct=("rating", lambda x: (x == 5).mean()),

            user_rating_skew=("rating", lambda x: skew(x)),
            user_rating_kurt=("rating", lambda x: kurtosis(x)),
            rating_entropy=("rating", lambda x: entropy(x.value_counts(normalize=True)))
        ).reset_index()

        # fill missing values
        user_features["std_rating"] = user_features["std_rating"].fillna(0)
        user_features["user_rating_skew"] = user_features["user_rating_skew"].fillna(0)
        user_features["user_rating_kurt"] = user_features["user_rating_kurt"].fillna(0)

        # derived feature
        user_features["interaction_density"] = user_features["num_unique_items_rated"] / 1000

        return user_features


class InteractionFeatureBuilder:
    def __init__(self):
        self.item_mean = None
        self.item_std = None
        self.item_popularity = None

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        # compute item-level statistics (TRAIN ONLY)
        self.item_popularity = df.groupby("item").size()
        self.item_mean_rating = df.groupby("item")["rating"].mean()
        self.item_std_rating = df.groupby("item")["rating"].std()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # map item stats
        df["item_popularity"] = df["item"].map(self.item_popularity)
        df["item_mean_rating"] = df["item"].map(self.item_mean_rating)
        df["item_std_rating"] = df["item"].map(self.item_std_rating)

        # handle missing values (important for test)
        df["item_popularity"] = df["item_popularity"].fillna(0)
        df["item_mean_rating"] = df["item_mean_rating"].fillna(df["rating"].mean())
        df["item_std_rating"] = df["item_std_rating"].fillna(1)

        # interaction features
        df["rating_deviation"] = df["rating"] - df["item_mean_rating"]
        df["normalized_deviation"] = df["rating_deviation"] / (df["item_std_rating"] + 1e-5)
        df["abs_deviation"] = df["rating_deviation"].abs()

        # directional signal
        df["negative_dev"] = (df["normalized_deviation"] < -1)
        df["positive_dev"] = (df["normalized_deviation"] > 1)

        # aggregate to user
        user_features = df.groupby("user").agg(
            mean_rating_deviation=("rating_deviation", "mean"),
            std_rating_deviation=("rating_deviation", "std"),
            mean_normalized_deviation=("normalized_deviation", "mean"),
            std_normalized_deviation=("normalized_deviation", "std"),
            mean_abs_deviation=("abs_deviation", "mean"),
            avg_item_popularity=("item_popularity", "mean"),

            max_abs_deviation=("abs_deviation", "max"),
            p90_abs_deviation=("abs_deviation", lambda x: np.percentile(x, 90)),
    
            negative_dev_ratio=("negative_dev", "mean"),
            positive_dev_ratio=("positive_dev", "mean"),
        ).reset_index()

        return user_features