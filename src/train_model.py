import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLUMNS = [
    "HOME",
    "days_rest",
    "pts_last3",
    "pts_last5",
    "pts_last10",
    "min_last3",
    "min_last5",
    "fga_last3",
    "fga_last5",
    "fg3a_last5",
    "fta_last5",
    "reb_last5",
    "ast_last5",
    "pts_std_last5",
    "pts_trend",
    "min_trend",
    "fga_trend",
]

TARGET_COLUMN = "PTS"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df


def split_data_by_date(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Split dataset by date instead of row index.
    Older games go to train set, newer games go to test set.
    """
    split_date = df["GAME_DATE"].quantile(train_ratio)

    train_df = df[df["GAME_DATE"] <= split_date].copy()
    test_df = df[df["GAME_DATE"] > split_date].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test, test_df, split_date


def evaluate_model(model_name: str, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def train_and_save_models(data_path: str):
    df = load_data(data_path)

    X_train, X_test, y_train, y_test, df_test, split_date = split_data_by_date(df)

    print(f"Train max date: {df[df['GAME_DATE'] <= split_date]['GAME_DATE'].max().date()}")
    print(f"Test min date:  {df[df['GAME_DATE'] > split_date]['GAME_DATE'].min().date()}")
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    results = []

    baseline_pred = X_test["pts_last5"]
    results.append(evaluate_model("baseline_last5", y_test, baseline_pred))

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    results.append(evaluate_model("linear_regression", y_test, lr_pred))

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results.append(evaluate_model("random_forest", y_test, rf_pred))

    with open("models/linear_regression.pkl", "wb") as f:
        pickle.dump(lr_model, f)

    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_results.csv", index=False)

    prediction_df = df_test[["PLAYER_NAME", "GAME_DATE", "MATCHUP", "PTS"]].copy()
    prediction_df["baseline_pred"] = baseline_pred.values
    prediction_df["lr_pred"] = lr_pred
    prediction_df["rf_pred"] = rf_pred
    prediction_df["baseline_error"] = prediction_df["PTS"] - prediction_df["baseline_pred"]
    prediction_df["lr_error"] = prediction_df["PTS"] - prediction_df["lr_pred"]
    prediction_df["rf_error"] = prediction_df["PTS"] - prediction_df["rf_pred"]

    prediction_df.to_csv("models/test_predictions.csv", index=False)

    print("\nModel results:")
    print(results_df.sort_values("mae"))

    print("\nSaved:")
    print("- models/model_results.csv")
    print("- models/test_predictions.csv")
    print("- models/linear_regression.pkl")
    print("- models/random_forest.pkl")


if __name__ == "__main__":
    train_and_save_models("data/processed/nba_players_model_dataset.csv")