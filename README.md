# DAX Stock Prediction Model

This project implements a stock price prediction model for the German DAX (Deutscher Aktienindex) using machine learning and time series forecasting techniques. The project compares Random Forest and ARIMA models for predicting future stock prices.

## Features

- Fetches historical DAX stock data using yfinance
- Creates various technical indicators for feature engineering
- Implements two prediction models:
  - Random Forest Regressor (machine learning approach)
  - ARIMA (statistical time series approach)
- Evaluates model performance using standard metrics (MSE, RMSE, MAE, R²)
- Generates predictions for future stock prices
- Visualizes results with detailed plots and charts

## Requirements

The project requires Python 3.10+ and the following packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- yfinance
- statsmodels

All dependencies are listed in the `requirements.txt` file.

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```
   python3 -m venv venv
   ```
3. Activate the virtual environment:
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
   - On Windows:
     ```
     venv\Scripts\activate
     ```
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script to fetch data, train models, and generate predictions:

```bash
python dax_stock_prediction.py
```

The script will:
1. Download the latest 5 years of DAX historical data
2. Preprocess the data and create technical indicators
3. Train both Random Forest and ARIMA models
4. Evaluate model performance on test data
5. Generate predictions for the next 30 trading days
6. Create visualizations saved as PNG files:
   - `dax_predictions.png`: Contains historical data, test predictions, and future forecasts
   - `feature_importance.png`: Shows the importance of each feature in the Random Forest model

## Model Description

### Random Forest
The Random Forest model uses multiple technical indicators as features:
- Simple Moving Averages (5, 20, and 50-day)
- Exponential Moving Averages (12 and 26-day)
- Daily, 5-day, and 20-day returns
- Price momentum
- Volatility

### ARIMA
The ARIMA (AutoRegressive Integrated Moving Average) model uses a statistical approach for time series forecasting with parameters:
- p=5 (autoregressive order)
- d=1 (differencing order)
- q=0 (moving average order)

## Disclaimer

This project is for educational purposes only. Stock market predictions are inherently uncertain, and no model can consistently predict future stock prices with high accuracy. Do not use these predictions for real investment decisions.

## License

This project is open source and available under the MIT License. 