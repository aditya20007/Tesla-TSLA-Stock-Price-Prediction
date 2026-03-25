# 🚗 Tesla Stock Price Prediction — SimpleRNN & LSTM

## Project Overview
Predict Tesla's adjusted closing stock price using Deep Learning (SimpleRNN and LSTM) for 1-day, 5-day, and 10-day horizons.

## Folder Structure
```
TSLA_Project/                        ← GitHub Repo Root
├── app/
│   └── streamlit_app.py             ← Streamlit Web App
├── data/
│   └── TSLA.csv                     ← Dataset (2416 rows)
├── notebooks/
│   └── TSLA_Stock_Prediction.ipynb  ← Main Jupyter Notebook
├── src/
│   ├── data_preprocessing.py        ← Data loading, cleaning, sequences
│   ├── model_builder.py             ← RNN & LSTM architectures
│   ├── train.py                     ← Training pipeline + GridSearchCV
│   └── evaluate.py                  ← Metrics and plots
├── models/                          ← Saved .h5 model files (after training)
├── reports/                         ← Saved plots (after training)
├── .python-version                  ← Forces Python 3.11 on Streamlit Cloud
├── requirements.txt                 ← All dependencies
└── README.md
```

## How to Run Locally
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open Jupyter Notebook
jupyter notebook notebooks/TSLA_Stock_Prediction.ipynb

# 3. Train models (from src/ folder)
cd src
python train.py

# 4. Evaluate models
python evaluate.py

# 5. Run Streamlit App
streamlit run app/streamlit_app.py
```

## Models Used
- **SimpleRNN** — Basic Recurrent Neural Network
- **LSTM** — Long Short-Term Memory Network

## Forecast Horizons
- 1 Day ahead
- 5 Days ahead
- 10 Days ahead

## Tech Stack
Python | TensorFlow/Keras | Scikit-learn | Pandas | Streamlit | Matplotlib | Seaborn
