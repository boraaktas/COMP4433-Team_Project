# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore


# CONSTANTS
OUTLIERS_TO_REMOVE = {"LotFrontage": 200, "LotArea": 100000, "GrLivArea": 1234}


# HELPING FUNCTIONS
def read_data(path: str = "data/train.csv", index_col: str = "Id") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=index_col)
    return df


def get_categorical_features(df: pd.DataFrame) -> list[str]:
    categorical_features = df.select_dtypes(include=["object"]).columns
    return categorical_features


def get_numerical_features(df: pd.DataFrame) -> list[str]:
    numerical_features = df.select_dtypes(include=[np.number]).columns
    return numerical_features


# HANDLE MISSING VALUES
def make_same_numtype(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    for col in test_df.columns:
        if np.issubdtype(test_df[col].dtype, np.number):
            if test_df[col].dtype != train_df[col].dtype:
                # print(f"Column: {col}, Test dtype: {test_df[col].dtype}, Train dtype: {train_df[col].dtype}")
                test_df[col] = test_df[col].fillna(0)
                test_df[col] = test_df[col].astype(int)
    return train_df, test_df


def calculate_NaN(df: pd.DataFrame, as_percentage: bool = False):
    if as_percentage:
        NaN_df = df.isna().mean() * 100

    else:
        NaN_df = df.isna().sum()
    features_with_missing = NaN_df[NaN_df > 0].sort_values(ascending=False)
    return NaN_df, features_with_missing


def fill_with_NaNstring(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df[features] = df[features].fillna("NaN")
    return df


def drop_features_with_high_missing(df: pd.DataFrame, limit: float) -> pd.DataFrame:
    _, features_missing = calculate_NaN(df, as_percentage=True)
    for feature in features_missing.index:
        if features_missing[feature] > limit:
            print("DROP", features_missing[feature])
            df = df.drop([feature], axis=1)
    return df


# Outliers for numerical features is explored in dataset_overview.ipynb.
# The outliers can be removed based on what we see in the plots (with a defined treshhold)
# or we can make another function that removes based on quantile
def remove_outliers_2(
    df: pd.DataFrame, features: dict = OUTLIERS_TO_REMOVE
) -> pd.DataFrame:
    """Removes outliers from the DataFrame for specified features based on given thresholds.

    Args:
        df (pd.DataFrame): The input DataFrame from which outliers need to be removed
        features (dict): A dictionary where keys are feature names (columns) and values are the threshold values.
                         Rows where the feature value exceeds the threshold will be removed.

    Returns:
        pd.DataFrame: A DataFrame with outliers removed based on the specified thresholds.
    """
    for feature, treshhold in features.items():
        df = df[df[feature] < treshhold]
    return df


def build_predicting_num_model(
    df: pd.DataFrame, model_name: str, features_to_use: list
):
    pass


def normalize_data(df, features: list[str]) -> pd.DataFrame:
    pass


def main():
    train_df = read_data()
    test_df = read_data("data/test.csv")
    train_df, test_df = make_same_numtype(train_df, test_df)
    drop_features_with_high_missing(train_df, 80)
    NaN_df, features_with_missing = calculate_NaN(train_df, True)
    print(features_with_missing)
    train_df = fill_with_NaNstring(train_df, get_categorical_features(train_df))


main()
