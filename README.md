# NBA Player Props Prediction System

Machine learning project for predicting NBA player scoring relative to market lines using historical game data, feature engineering, and model comparison.

---

## Project Overview

This project analyzes NBA player performance and predicts the number of points a player is likely to score in the next game. The prediction can then be compared with a market line (for example: **26.5 points**) to generate an **over/under suggestion**.

The current **MVP version** focuses on **LeBron James** and uses historical game logs from the NBA API.

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
│   ├── features.py
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

## Future Improvements

- Support for **multiple NBA players**
- **Full-league dataset**
- **Bookmaker market line integration**
- Additional features such as **opponent defensive strength**
- **XGBoost / LightGBM models**
- Better **evaluation and backtesting**
- Deployment as a **public web application**

---

## Disclaimer

This project is for **educational and portfolio purposes only**.  
It is **not financial advice**.
