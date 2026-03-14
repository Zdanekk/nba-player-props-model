import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic rolling and contextual features for player points prediction.
    """
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    if "HOME" not in df.columns:
        df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

    if "WIN" not in df.columns:
        df["WIN"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)

    numeric_cols = ["MIN", "FGA", "FG3A", "FTA", "REB", "AST", "TOV", "PTS"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pts_last3"] = df["PTS"].shift(1).rolling(3).mean()
    df["pts_last5"] = df["PTS"].shift(1).rolling(5).mean()
    df["pts_last10"] = df["PTS"].shift(1).rolling(10).mean()

    df["min_last3"] = df["MIN"].shift(1).rolling(3).mean()
    df["min_last5"] = df["MIN"].shift(1).rolling(5).mean()

    df["fga_last3"] = df["FGA"].shift(1).rolling(3).mean()
    df["fga_last5"] = df["FGA"].shift(1).rolling(5).mean()

    df["fg3a_last5"] = df["FG3A"].shift(1).rolling(5).mean()
    df["fta_last5"] = df["FTA"].shift(1).rolling(5).mean()

    df["reb_last5"] = df["REB"].shift(1).rolling(5).mean()
    df["ast_last5"] = df["AST"].shift(1).rolling(5).mean()

    df["pts_std_last5"] = df["PTS"].shift(1).rolling(5).std()

    df["days_rest"] = df["GAME_DATE"].diff().dt.days
    df["days_rest"] = df["days_rest"].fillna(3)

    return df


def prepare_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare final dataset for modeling.
    """
    df = add_basic_features(df)

    feature_columns = [
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
    ]

    model_df = df[["GAME_DATE", "MATCHUP", "PTS"] + feature_columns].copy()
    model_df = model_df.dropna().reset_index(drop=True)

    return model_df