# NBA Player Props Prediction System

A machine learning project that predicts NBA player scoring and evaluates over/under decisions using historical game data and feature engineering.

---

## Project Overview

This project analyzes NBA player performance and predicts the number of points a player is likely to score in the next game. The prediction can then be compared with a market line (for example: **26.5 points**) to generate an **over/under suggestion**.

The current **MVP version** supports multiple **NBA players** and uses historical game logs from the NBA API.

---

## Features

- Data collection from **NBA API**
- Feature engineering based on **recent form and game context**
- Model training and evaluation
- Next-game prediction
- Simple **Streamlit dashboard**
- Over/under decision support

---

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- nba_api
- Streamlit

---

## Project Structure

```text
nba-player-props-model
│
├── data
│   ├── raw
│   └── processed
│
├── notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src
│   ├── data_loader.py
│   ├── train_model.py
│   ├── predict.py
│   └── evaluation.py
│
├── app
│   └── streamlit_app.py
│
├── models
│
├── README.md
└── requirements.txt
```

---

## Data Source

The project uses the **nba_api** package to collect NBA player game logs.

---

## Current Modeling Approach

The current version uses:

- **Baseline model** (last 5 games average)
- **Linear Regression**
- **Random Forest Regressor**

### Target

- Player **points scored in a game**

---

## Model Evaluation

Models are evaluated using a **time-based train/test split**, ensuring that training uses historical games while testing uses more recent games.

Evaluation metrics include:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² score**

The best performing models achieved approximately:

- MAE ≈ **4.8 points**
- RMSE ≈ **6.3 points**

### Over / Under Backtest

To evaluate the model from a decision-making perspective, a simple backtest was implemented.

A simulated line was created using the player's recent scoring average (`pts_last10`).  
Predictions were compared against this line to generate **OVER / UNDER decisions**.

Backtest result:

**≈ 54% prediction accuracy**

While this is only a proxy for real bookmaker lines, it demonstrates that the model captures meaningful signals in player performance trends.

---

## Example Features

- Home / away
- Days of rest
- Points average over last **3, 5, and 10 games**
- Minutes average
- Field goal attempts average
- 3-point attempts average
- Free throw attempts average
- Rebounds and assists average
- Recent scoring volatility

---

## Example Output

- Predicted **points for next game**
- **Over / Under suggestion** relative to selected line

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download player data

```bash
python src/data_loader.py
```

### 3. Create processed dataset

Use the notebooks:

- `01_data_collection.ipynb`
- `02_feature_engineering.ipynb`

### 4. Train models

```bash
python src/train_model.py
```

### 5. Generate prediction

```bash
python src/predict.py
```

### 6. Run Streamlit app

```bash
python -m streamlit run app/streamlit_app.py
```

---

## Example Workflow

1. Collect NBA player game logs from the NBA API
2. Generate rolling and trend-based features
3. Train regression models to predict player points
4. Evaluate models using a time-based split
5. Compare predictions against a simulated scoring line
6. Generate over/under suggestions
## Future Improvements

---

Planned improvements for future iterations:

- Integration with **real bookmaker market lines via API**
- Additional contextual features (opponent defensive rating, team pace)
- Advanced models such as **XGBoost or LightGBM**
- Automated daily predictions for upcoming games
- Improved decision metrics and backtesting
- Deployment as a **public web application**

---

## Disclaimer

This project is for **educational and portfolio purposes only**.  
It is **not financial advice**.
