import pandas as pd
import pickle


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
]


def load_model():
    with open("models/random_forest.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def load_dataset():
    return pd.read_csv("data/processed/nba_players_model_dataset.csv")


def predict_for_player(player_name: str, line: float = 20.5):
    df = load_dataset()

    player_df = df[df["PLAYER_NAME"] == player_name].copy()

    if player_df.empty:
        raise ValueError(f"No data found for player: {player_name}")

    player_df = player_df.sort_values("GAME_DATE")
    last_row = player_df.iloc[[-1]][FEATURE_COLUMNS].copy()

    model = load_model()
    prediction = model.predict(last_row)[0]

    decision = "OVER" if prediction > line else "UNDER"

    return {
        "player_name": player_name,
        "predicted_points": round(prediction, 2),
        "line": line,
        "decision": decision
    }


if __name__ == "__main__":
    result = predict_for_player("Alex Caruso", line=26.5)

    print("Player:", result["player_name"])
    print("Predicted points:", result["predicted_points"])
    print("Line:", result["line"])
    print("Model suggests:", result["decision"])