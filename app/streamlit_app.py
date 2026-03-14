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
DATA_PATH = BASE_DIR / "data" / "processed" / "lebron_model_dataset.csv"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


st.title("NBA Player Props Prediction")
st.subheader("LeBron James - Points Prediction")

df = pd.read_csv(DATA_PATH)
model = load_model()

line = st.number_input("Points line", value=26.5, step=0.5)

last_row = df.iloc[[-1]][FEATURE_COLUMNS].copy()
prediction = model.predict(last_row)[0]

st.metric("Predicted points", round(prediction, 2))

if prediction > line:
    st.success(f"Model suggests OVER {line}")
else:
    st.error(f"Model suggests UNDER {line}")

st.write("Latest input features used for prediction:")
st.dataframe(last_row)