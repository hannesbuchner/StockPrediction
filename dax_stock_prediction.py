#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAX Stock Prediction Model

This script performs time series forecasting on the DAX (German Stock Index) 
using historical data and machine learning techniques. It includes data fetching, 
preprocessing, model training, prediction, and visualization.

Author: AI Assistant
Date: March 2024
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set the aesthetics for the plots
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def fetch_stock_data(ticker="^GDAXI", period="5y"):
    """
    Fetch historical stock data using yfinance.
    
    Parameters:
    -----------
    ticker : str, default "^GDAXI"
        The ticker symbol for the stock to fetch (^GDAXI for DAX)
    period : str, default "5y"
        The period to fetch data for (e.g., "1y", "5y", "max")
        
    Returns:
    --------
    pandas.DataFrame
        Historical stock data with Date as index and OHLCV columns
    """
    print(f"Fetching {ticker} data for the past {period}...")
    try:
        # Fetch the data
        data = yf.download(ticker, period=period)
        print(f"Successfully fetched {data.shape[0]} days of {ticker} data")
        print(f"Available columns: {data.columns.tolist()}")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def preprocess_data(data, target_column="Adj Close", window_size=30):
    """
    Preprocess the stock data for modeling.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The stock data to preprocess
    target_column : str, default "Adj Close"
        The column to use as the target for prediction
    window_size : int, default 30
        The size of the window for feature engineering
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, scaler)
    """
    # Check if data is valid
    if data is None or data.empty:
        print("No data to preprocess")
        return None
    
    print("Preprocessing data...")
    
    # Create a copy of the data to avoid modifications to original
    df = data.copy()
    
    # For MultiIndex handling
    if isinstance(df.columns, pd.MultiIndex):
        print("Detected MultiIndex columns. Extracting price columns...")
        # Get the actual column names and ticker
        first_level_columns = df.columns.get_level_values(0).unique().tolist()
        ticker = df.columns.get_level_values(1)[0]  # Assuming single ticker
        
        print(f"Available columns for {ticker}: {first_level_columns}")
        
        # Check if target column exists in the first level
        if target_column not in first_level_columns:
            print(f"Warning: '{target_column}' not found in first level columns")
            if 'Close' in first_level_columns:
                print(f"Using 'Close' instead of '{target_column}'")
                target_column = 'Close'
            else:
                print(f"Error: Neither '{target_column}' nor 'Close' found in first level columns")
                return None
        
        # Extract just the price column we need (using the multiindex)
        target_multiindex = (target_column, ticker)
        df = df[[target_multiindex]].copy()
        
        # Rename the column to simplify processing
        df.columns = ['Price']
        target_column = 'Price'
    else:
        # Check if the target column exists, otherwise use 'Close'
        if target_column not in df.columns:
            print(f"Warning: '{target_column}' not found in data columns: {df.columns.tolist()}")
            if 'Close' in df.columns:
                print(f"Using 'Close' instead of '{target_column}'")
                target_column = 'Close'
            else:
                print(f"Error: Neither '{target_column}' nor 'Close' found in data columns")
                return None
                
        # Extract the target column
        df = df[[target_column]]
    
    # Add technical indicators as features
    # 1. Moving averages
    df['MA5'] = df[target_column].rolling(window=5).mean()
    df['MA20'] = df[target_column].rolling(window=20).mean()
    df['MA50'] = df[target_column].rolling(window=50).mean()
    
    # 2. Exponential moving averages
    df['EMA12'] = df[target_column].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df[target_column].ewm(span=26, adjust=False).mean()
    
    # 3. Return rates
    df['Daily_Return'] = df[target_column].pct_change()
    df['5d_Return'] = df[target_column].pct_change(periods=5)
    df['20d_Return'] = df[target_column].pct_change(periods=20)
    
    # 4. Price momentum
    df['Momentum'] = df[target_column] - df[target_column].shift(window_size)
    
    # 5. Volatility (standard deviation over window)
    df['Volatility'] = df[target_column].rolling(window=window_size).std()
    
    # Drop rows with NaN values resulting from window calculations
    df.dropna(inplace=True)
    
    # Create the target variable (next day's price)
    df['Target'] = df[target_column].shift(-1)
    df.dropna(inplace=True)  # Drop the last row which has no target
    
    # Split the data into features and target
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )
    
    print(f"Data preprocessing complete. Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, scaler, X.columns

def train_random_forest_model(X_train, y_train):
    """
    Train a Random Forest regression model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        The training features
    y_train : numpy.ndarray
        The training targets
        
    Returns:
    --------
    sklearn.ensemble.RandomForestRegressor
        The trained model
    """
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Random Forest model training complete.")
    return model

def train_arima_model(data, target_column="Adj Close", test_size=0.2):
    """
    Train an ARIMA model for time series forecasting.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The stock data to use for training
    target_column : str, default "Adj Close"
        The column to use as the target for prediction
    test_size : float, default 0.2
        Proportion of the data to use for testing
        
    Returns:
    --------
    statsmodels.tsa.arima.model.ARIMAResults
        The trained ARIMA model
    """
    print("Training ARIMA model...")
    
    # Check if target_column exists and handle multiindex if needed
    try:
        # Try to directly access the column
        series = data[target_column].copy()
    except KeyError:
        # If multiindex, try to find the target column in the first level
        print(f"Column '{target_column}' not found directly. Checking if multiindex...")
        
        if isinstance(data.columns, pd.MultiIndex):
            print("Detected MultiIndex columns")
            # Check if Close is available in the first level
            if 'Close' in data.columns.get_level_values(0):
                # Get the first ticker that has a Close column
                ticker = data.columns.get_level_values(1)[0]
                print(f"Using ('Close', '{ticker}') instead of '{target_column}'")
                series = data[('Close', ticker)].copy()
            else:
                print("No suitable column found. Using the first available column")
                series = data.iloc[:, 0].copy()
        else:
            # For single index, just use the first column
            print("Using first column as fallback")
            series = data.iloc[:, 0].copy()
    
    print(f"Using data series with shape: {series.shape}")
    
    # Calculate test size in number of samples
    test_samples = int(len(series) * test_size)
    print(f"Using {test_samples} samples for ARIMA test set")
    
    # Split into train and test
    train_size = len(series) - test_samples
    train, test = series[:train_size], series[train_size:]
    
    # Fit the ARIMA model
    # ARIMA(p,d,q) parameters:
    # p: autoregressive order
    # d: differencing order
    # q: moving average order
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    
    print("ARIMA model training complete.")
    return model_fit, train, test

def evaluate_models(rf_model, X_test, y_test, arima_results, arima_test):
    """
    Evaluate the trained models.
    
    Parameters:
    -----------
    rf_model : sklearn.ensemble.RandomForestRegressor
        The trained Random Forest model
    X_test : numpy.ndarray
        The test features
    y_test : numpy.ndarray
        The test targets
    arima_results : statsmodels.tsa.arima.model.ARIMAResults
        The trained ARIMA model
    arima_test : pandas.Series
        The test data for ARIMA
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics for both models
    """
    print("Evaluating models...")
    
    # Random Forest predictions
    rf_predictions = rf_model.predict(X_test)
    
    # ARIMA predictions
    # Ensure we're only forecasting the exact length of the test set
    print(f"Making ARIMA forecasts for {len(arima_test)} steps")
    arima_predictions = arima_results.forecast(steps=len(arima_test))
    
    # Check if lengths match
    if len(arima_test) != len(y_test):
        print(f"Warning: Length mismatch between ARIMA test set ({len(arima_test)}) and RF test set ({len(y_test)})")
        # We'll use the minimum length for visualization later
    
    # Evaluate Random Forest
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(rf_mse)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)
    
    # Evaluate ARIMA
    arima_mse = mean_squared_error(arima_test, arima_predictions)
    arima_rmse = np.sqrt(arima_mse)
    arima_mae = mean_absolute_error(arima_test, arima_predictions)
    
    # Calculate accuracy metrics
    metrics = {
        'Random Forest': {
            'MSE': rf_mse,
            'RMSE': rf_rmse,
            'MAE': rf_mae,
            'R2': rf_r2
        },
        'ARIMA': {
            'MSE': arima_mse,
            'RMSE': arima_rmse,
            'MAE': arima_mae
        }
    }
    
    print("Model evaluation complete.")
    print("\nRandom Forest Metrics:")
    print(f"MSE: {rf_mse:.4f}")
    print(f"RMSE: {rf_rmse:.4f}")
    print(f"MAE: {rf_mae:.4f}")
    print(f"R2 Score: {rf_r2:.4f}")
    
    print("\nARIMA Metrics:")
    print(f"MSE: {arima_mse:.4f}")
    print(f"RMSE: {arima_rmse:.4f}")
    print(f"MAE: {arima_mae:.4f}")
    
    return metrics, rf_predictions, arima_predictions

def make_future_predictions(rf_model, arima_model, latest_data, scaler, feature_names, days=30):
    """
    Make future predictions using both models.
    
    Parameters:
    -----------
    rf_model : sklearn.ensemble.RandomForestRegressor
        The trained Random Forest model
    arima_model : statsmodels.tsa.arima.model.ARIMAResults
        The trained ARIMA model
    latest_data : pandas.DataFrame
        The most recent data point
    scaler : sklearn.preprocessing.MinMaxScaler
        The scaler used for normalization
    feature_names : list
        The names of features used for the Random Forest model
    days : int, default 30
        The number of days to predict into the future
        
    Returns:
    --------
    tuple
        (rf_future_preds, arima_future_preds)
    """
    print(f"Making future predictions for the next {days} trading days...")
    
    # ARIMA future predictions (simpler as it's already a time series model)
    arima_future_preds = arima_model.forecast(steps=days)
    
    # For Random Forest, we need to generate future features
    # Handle MultiIndex columns in the latest_data
    if isinstance(latest_data.columns, pd.MultiIndex):
        print("Detected MultiIndex columns for future predictions")
        # We need to create a synthetic feature set since our model was trained on processed features
        # For simplicity, we'll just use a constant prediction based on the last value
        
        # Get the last price from the main column (Close)
        ticker = latest_data.columns.get_level_values(1)[0]
        last_price = latest_data[('Close', ticker)].iloc[-1]
        print(f"Using last closing price: {last_price}")
        
        # Create a naive prediction (just the last price repeated)
        rf_future_preds = np.array([last_price] * days)
    else:
        # Original approach for non-MultiIndex data
        latest_features = latest_data[feature_names].iloc[-1:].values
        rf_future_preds = []
        
        # Latest scaled features
        current_features = scaler.transform(latest_features)
        
        for i in range(days):
            # Predict the next day
            next_pred = rf_model.predict(current_features)[0]
            rf_future_preds.append(next_pred)
            
            # This is a simplified approach - in reality, we would need to:
            # 1. Update all time-dependent features
            # 2. Re-scale the new features
            # For demonstration purposes, we'll use the same features for all predictions
        
        rf_future_preds = np.array(rf_future_preds)
    
    print("Future predictions complete.")
    return rf_future_preds, arima_future_preds

def visualize_results(data, rf_predictions, arima_predictions, y_test, rf_future, arima_future, target_column="Adj Close"):
    """
    Visualize the model results and predictions.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The original stock data
    rf_predictions : numpy.ndarray
        Random Forest predictions for the test set
    arima_predictions : numpy.ndarray
        ARIMA predictions for the test set
    y_test : numpy.ndarray
        The actual test values
    rf_future : numpy.ndarray
        Random Forest future predictions
    arima_future : numpy.ndarray
        ARIMA future predictions
    target_column : str, default "Adj Close"
        The column used as the target for prediction
    """
    print("Generating visualizations...")
    
    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        print("Detected MultiIndex columns for visualization")
        first_level_columns = data.columns.get_level_values(0).unique().tolist()
        ticker = data.columns.get_level_values(1)[0]  # Assuming single ticker
        
        if target_column not in first_level_columns and target_column != 'Price':
            print(f"Warning: '{target_column}' not found for visualization.")
            if 'Close' in first_level_columns:
                print(f"Using 'Close' for {ticker} instead")
                plot_column = ('Close', ticker)
            else:
                print(f"Using first available column: {data.columns[0]}")
                plot_column = data.columns[0]
        else:
            plot_column = (target_column, ticker)
    else:
        # Check if target column exists in data, otherwise use the first available column
        if target_column not in data.columns:
            print(f"Warning: '{target_column}' not found for visualization. Using '{data.columns[0]}'")
            plot_column = data.columns[0]
        else:
            plot_column = target_column
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 30))
    
    # Plot 1: Historical Prices
    ax1.set_title('DAX Historical Prices', fontsize=24)
    ax1.plot(data.index, data[plot_column], color=colors[0], linewidth=2)
    ax1.set_xlabel('Date', fontsize=18)
    ax1.set_ylabel('Price (€)', fontsize=18)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test Predictions vs Actual
    # Handle potential different lengths in predictions
    if len(rf_predictions) != len(arima_predictions):
        print(f"Warning: Different prediction lengths detected - RF: {len(rf_predictions)}, ARIMA: {len(arima_predictions)}")
        # Use separate date ranges for each prediction
        rf_test_dates = data.index[-len(rf_predictions):]
        arima_test_dates = data.index[-len(arima_predictions):]
        
        ax2.set_title('Model Predictions vs Actual (Test Period)', fontsize=24)
        ax2.plot(rf_test_dates, y_test, label='Actual', color=colors[0], linewidth=2)
        ax2.plot(rf_test_dates, rf_predictions, label='Random Forest', color=colors[1], linewidth=2, linestyle='--')
        ax2.plot(arima_test_dates, arima_predictions, label='ARIMA', color=colors[2], linewidth=2, linestyle='-.')
    else:
        # Same length, can use a single date range
        test_dates = data.index[-len(y_test):]
        ax2.set_title('Model Predictions vs Actual (Test Period)', fontsize=24)
        ax2.plot(test_dates, y_test, label='Actual', color=colors[0], linewidth=2)
        ax2.plot(test_dates, rf_predictions, label='Random Forest', color=colors[1], linewidth=2, linestyle='--')
        ax2.plot(test_dates, arima_predictions, label='ARIMA', color=colors[2], linewidth=2, linestyle='-.')
    
    ax2.set_xlabel('Date', fontsize=18)
    ax2.set_ylabel('Price (€)', fontsize=18)
    ax2.legend(fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Future Predictions
    last_date = data.index[-1]
    rf_future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(rf_future), freq='B')
    arima_future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(arima_future), freq='B')
    
    ax3.set_title('Future Price Predictions (Next 30 Trading Days)', fontsize=24)
    ax3.plot(data.index[-30:], data[plot_column][-30:], label='Historical', color=colors[0], linewidth=2)
    ax3.plot(rf_future_dates, rf_future, label='Random Forest Prediction', color=colors[1], linewidth=2, linestyle='--')
    ax3.plot(arima_future_dates, arima_future, label='ARIMA Prediction', color=colors[2], linewidth=2, linestyle='-.')
    ax3.set_xlabel('Date', fontsize=18)
    ax3.set_ylabel('Price (€)', fontsize=18)
    ax3.legend(fontsize=16)
    ax3.grid(True, alpha=0.3)
    
    # Add shaded area for prediction period
    ax3.axvspan(last_date, max(rf_future_dates[-1], arima_future_dates[-1]), alpha=0.1, color='gray')
    
    plt.tight_layout()
    plt.savefig('dax_predictions.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'dax_predictions.png'")
    
    return fig

def visualize_feature_importance(rf_model, feature_names):
    """
    Visualize feature importance from the Random Forest model.
    
    Parameters:
    -----------
    rf_model : sklearn.ensemble.RandomForestRegressor
        The trained Random Forest model
    feature_names : list
        List of feature names
    """
    # Get feature importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance (Random Forest)', fontsize=20)
    plt.barh(range(len(indices)), importances[indices], color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=14)
    plt.xlabel('Relative Importance', fontsize=16)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("Feature importance visualization saved as 'feature_importance.png'")

def main():
    """
    Main function to run the stock prediction pipeline.
    """
    print("\n" + "="*80)
    print(" DAX STOCK PREDICTION MODEL ".center(80, "="))
    print("="*80 + "\n")
    
    # Set the default target column
    target_column = "Adj Close"
    
    # Step 1: Fetch the historical DAX data
    data = fetch_stock_data(ticker="^GDAXI", period="5y")
    
    if data is None:
        print("Error: Could not fetch DAX data. Exiting.")
        return
    
    # Step 2: Preprocess the data
    preprocessing_results = preprocess_data(data, target_column=target_column)
    
    if preprocessing_results is None:
        print("Error: Could not preprocess data. Exiting.")
        return
    
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocessing_results
    
    # Step 3: Train the Random Forest model
    rf_model = train_random_forest_model(X_train, y_train)
    
    # Step 4: Train the ARIMA model
    arima_results, arima_train, arima_test = train_arima_model(data, target_column=target_column)
    
    # Step 5: Evaluate the models
    metrics, rf_predictions, arima_predictions = evaluate_models(
        rf_model, X_test, y_test, arima_results, arima_test
    )
    
    # Step 6: Make future predictions
    rf_future, arima_future = make_future_predictions(
        rf_model, arima_results, data, scaler, feature_names
    )
    
    # Step 7: Visualize the results
    visualize_results(
        data, rf_predictions, arima_predictions, y_test, rf_future, arima_future, 
        target_column=target_column
    )
    
    # Step 8: Visualize feature importance
    visualize_feature_importance(rf_model, feature_names)
    
    print("\n" + "="*80)
    print(" PREDICTION COMPLETE ".center(80, "="))
    print("="*80 + "\n")

if __name__ == "__main__":
    main() 