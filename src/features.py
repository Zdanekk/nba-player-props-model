import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df = df.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)

    if "HOME" not in df.columns:
        df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

    if "WIN" not in df.columns:
        df["WIN"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)

    numeric_cols = ["MIN", "FGA", "FG3A", "FTA", "REB", "AST", "TOV", "PTS"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby("PLAYER_NAME", group_keys=False)

    df["pts_last3"] = grouped["PTS"].transform(lambda s: s.shift(1).rolling(3).mean())
    df["pts_last5"] = grouped["PTS"].transform(lambda s: s.shift(1).rolling(5).mean())
    df["pts_last10"] = grouped["PTS"].transform(lambda s: s.shift(1).rolling(10).mean())

    df["min_last3"] = grouped["MIN"].transform(lambda s: s.shift(1).rolling(3).mean())
    df["min_last5"] = grouped["MIN"].transform(lambda s: s.shift(1).rolling(5).mean())

    df["fga_last3"] = grouped["FGA"].transform(lambda s: s.shift(1).rolling(3).mean())
    df["fga_last5"] = grouped["FGA"].transform(lambda s: s.shift(1).rolling(5).mean())

    df["fg3a_last5"] = grouped["FG3A"].transform(lambda s: s.shift(1).rolling(5).mean())
    df["fta_last5"] = grouped["FTA"].transform(lambda s: s.shift(1).rolling(5).mean())

    df["reb_last5"] = grouped["REB"].transform(lambda s: s.shift(1).rolling(5).mean())
    df["ast_last5"] = grouped["AST"].transform(lambda s: s.shift(1).rolling(5).mean())

    df["pts_std_last5"] = grouped["PTS"].transform(lambda s: s.shift(1).rolling(5).std())

    df["days_rest"] = grouped["GAME_DATE"].diff().dt.days
    df["days_rest"] = df["days_rest"].fillna(3)

    return df


def prepare_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
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

    keep_columns = [
        "PLAYER_NAME",
        "PLAYER_ID",
        "GAME_DATE",
        "MATCHUP",
        "PTS"
    ] + feature_columns

    model_df = df[keep_columns].copy()
    model_df = model_df.dropna().reset_index(drop=True)

    return model_df