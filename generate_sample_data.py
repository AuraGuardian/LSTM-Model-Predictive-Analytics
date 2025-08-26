import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal=9):
    """Calculate MACD and Signal Line"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def calculate_adx(high, low, close, window=14):
    """Calculate Average Directional Index (ADX)"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (window - 1)) + dx) / window
    
    return adx, plus_di, minus_di

def generate_sample_data(n_days=1000, start_price=100.0, volatility=0.02, trend=0.0001):
    """
    Generate sample stock market data with realistic patterns and technical indicators
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate random walk for prices with trend
    returns = np.random.normal(0, volatility, n_days) + trend
    prices = start_price * (1 + returns).cumprod()
    
    # Add some seasonality
    seasonality = 5 * np.sin(np.linspace(0, 20*np.pi, n_days))
    prices += seasonality
    
    # Generate OHLCV data with more realistic patterns
    close_prices = np.round(prices, 2)
    open_prices = np.round(close_prices * (1 + np.random.normal(0, 0.002, n_days)), 2)
    high_prices = np.round(np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.008, n_days))), 2)
    low_prices = np.round(np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.008, n_days))), 2)
    volumes = np.random.lognormal(10, 1, n_days).astype(int)
    
    # Ensure all arrays have the same length
    min_length = min(len(dates), len(open_prices), len(high_prices), len(low_prices), len(close_prices), len(volumes))
    dates = dates[:min_length]
    open_prices = open_prices[:min_length]
    high_prices = high_prices[:min_length]
    low_prices = low_prices[:min_length]
    close_prices = close_prices[:min_length]
    volumes = volumes[:min_length]
    
    # Create initial DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.round(open_prices, 2),
        'High': np.round(high_prices, 2),
        'Low': np.round(low_prices, 2),
        'Close': close_prices,
        'Volume': volumes
    })
    
    # Calculate technical indicators
    # 1. RSI (Relative Strength Index)
    df['RSI'] = calculate_rsi(df['Close'])
    
    # 2. MACD and Signal Line
    df['MACD'], df['Signal_Line'] = calculate_macd(df['Close'])
    
    # 3. Bollinger Bands
    df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = calculate_bollinger_bands(df['Close'])
    
    # 4. ADX with DI+ and DI-
    df['ADX'], df['DI_Plus'], df['DI_Minus'] = calculate_adx(df['High'], df['Low'], df['Close'])
    
    # 5. Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 6. Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    # 7. Price Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=5)
    
    # 8. Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    # 9. Lagged features
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    
    # 10. Rolling statistics
    df['Rolling_Mean_5'] = df['Close'].rolling(window=5).mean()
    df['Rolling_Std_5'] = df['Close'].rolling(window=5).std()
    
    # Drop NaN values that result from indicator calculations
    df = df.dropna()
    
    # Save to CSV
    df.to_csv('sample_stock_data.csv', index=False)
    print(f"Generated {len(df)} days of sample stock data.")
    print(df.head())

if __name__ == "__main__":
    generate_sample_data()
