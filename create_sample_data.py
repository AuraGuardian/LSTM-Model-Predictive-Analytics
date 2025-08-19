import os
import pandas as pd
from datetime import datetime, timedelta
from polygon import RESTClient
import time

def fetch_aapl_data(api_key: str, years: int = 5) -> pd.DataFrame:
    """Fetch real AAPL stock data from Polygon.io"""
    client = RESTClient(api_key=api_key)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    print(f"Fetching AAPL data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Fetch the data
    aggs = []
    for a in client.list_aggs(
        ticker="AAPL",
        multiplier=1,
        timespan='day',
        from_=start_date.strftime('%Y-%m-%d'),
        to=end_date.strftime('%Y-%m-%d'),
        limit=50000
    ):
        aggs.append(a)
        
    if not aggs:
        raise ValueError("No data returned from Polygon.io")
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'Open': agg.open,
        'High': agg.high,
        'Low': agg.low,
        'Close': agg.close,
        'Volume': agg.volume,
        'Date': pd.to_datetime(agg.timestamp, unit='ms')
    } for agg in aggs])
    
    # Set Date as index and sort
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    return df

def generate_sample_data():
    """Generate sample data, trying Polygon.io first, then falling back to random data"""
    # Try to get API key from environment variable
    api_key = os.getenv('POLYGON_API_KEY')
    
    if api_key:
        try:
            print("Fetching real AAPL data from Polygon.io...")
            return fetch_aapl_data(api_key)
        except Exception as e:
            print(f"Error fetching from Polygon.io: {e}. Falling back to random data.")
    else:
        print("No POLYGON_API_KEY found in environment variables. Using random data.")
    
    # Fallback to random data if Polygon fetch fails or no API key
    print("Generating random sample data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    np.random.seed(42)
    n = len(dates)
    base = np.linspace(100, 200, n)
    noise = np.random.normal(0, 5, n)
    
    close_prices = base + noise
    open_prices = close_prices * (1 + np.random.normal(0, 0.005, n))
    high_prices = close_prices * (1 + np.random.uniform(0, 0.01, n))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.01, n))
    
    high_prices = np.maximum(high_prices, open_prices, close_prices)
    low_prices = np.minimum(low_prices, open_prices, close_prices)
    
    return pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, size=n)
    }, index=dates)

if __name__ == "__main__":
    # Generate and save the data
    try:
        data = generate_sample_data()
        output_file = "sample_aapl_data.xlsx"
        data.to_excel(output_file)
        print(f"AAPL data saved to {output_file}")
        print(f"Data range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"Number of data points: {len(data)}")
    except Exception as e:
        print(f"Error: {e}")
