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


def predict_next_game(data_path: str, line: float = 26.5):
    df = pd.read_csv(data_path)

    last_row = df.iloc[[-1]][FEATURE_COLUMNS].copy()

    model = load_model()
    prediction = model.predict(last_row)[0]

    decision = "OVER" if prediction > line else "UNDER"

    return prediction, decision


if __name__ == "__main__":
    pred, decision = predict_next_game("data/processed/lebron_model_dataset.csv", line=26.5)

    print("Predicted points for next game:", round(pred, 2))
    print("Model suggests:", decision)