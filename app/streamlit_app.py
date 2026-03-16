import streamlit as st
import pandas as pd
import pickle
from pathlib import Path


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

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "nba_players_model_dataset.csv"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


st.title("NBA Player Props Prediction")
st.subheader("Full League MVP")

df = load_data()
model = load_model()

players = sorted(df["PLAYER_NAME"].unique())
selected_player = st.selectbox("Select player", players)

default_line = 20.5
line = st.number_input("Points line", value=default_line, step=0.5)

player_df = df[df["PLAYER_NAME"] == selected_player].sort_values("GAME_DATE")

last_row_features = player_df.iloc[[-1]][FEATURE_COLUMNS].copy()
prediction = model.predict(last_row_features)[0]

st.metric("Predicted points", round(prediction, 2))

if prediction > line:
    st.success(f"Model suggests OVER {line}")
else:
    st.error(f"Model suggests UNDER {line}")

st.write("Latest player game context used for prediction:")
preview_cols = ["PLAYER_NAME", "GAME_DATE", "MATCHUP", "PTS"] + FEATURE_COLUMNS
st.dataframe(player_df[preview_cols].tail(5).sort_values("GAME_DATE", ascending=False))