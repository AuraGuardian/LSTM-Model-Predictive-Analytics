import numpy as np
import pandas as pd
import os

def generate_sample_data():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='B')  # Business days
    
    # Generate synthetic stock prices (random walk)
    n_days = len(dates)
    returns = np.random.normal(0.0005, 0.01, n_days)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Starting price of 100
    
    # Add some seasonality
    seasonality = 5 * np.sin(np.linspace(0, 20*np.pi, n_days))
    prices += seasonality
    
    # Generate OHLCV data
    close_prices = np.round(prices, 2)
    open_prices = close_prices * (1 + np.random.normal(0, 0.005, n_days))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    volumes = np.random.lognormal(10, 1, n_days).astype(int)
    
    # Create DataFrame with all required columns
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.round(open_prices, 2),
        'High': np.round(high_prices, 2),
        'Low': np.round(low_prices, 2),
        'Close': close_prices,
        'Volume': volumes
    })
    
    # Save to CSV
    df.to_csv('sample_stock_data.csv', index=False)
    print(f"Generated {len(df)} days of sample stock data.")
    print(df.head())

if __name__ == "__main__":
    generate_sample_data()
