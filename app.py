import os
import time
import threading
import uvicorn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any
from dotenv import load_dotenv
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates

# Try to import yfinance with a fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("yfinance is not installed. Some features may be limited.")

# Import local modules
try:
    from stock_prediction import StockPredictor, load_sample_data
except ImportError as e:
    st.error(f"Error importing local modules: {e}")
    raise

# Import Polygon client with error handling
try:
    from polygon import RESTClient
except ImportError:
    RESTClient = None
    st.warning("Polygon API client not available. Some features may be limited.")

# Import monitoring
from monitoring import monitor
from metrics_server import start_metrics_server

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Initialize session state variables
if 'polygon_client' not in st.session_state:
    st.session_state.polygon_client = None
    st.session_state.use_sample_data = True  # Default to using sample data

# Function to initialize Polygon client
def init_polygon_client():
    try:
        if 'POLYGON_API_KEY' not in os.environ:
            return None
        return RESTClient(api_key=os.environ['POLYGON_API_KEY'])
    except Exception as e:
        st.error(f"Error initializing Polygon client: {str(e)}")
        return None

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["LSTM Prediction", "SARIMA Analysis"]
page = st.sidebar.radio("Go to", pages, index=0)

# Sidebar for stock selection
with st.sidebar:
    st.subheader("Stock Selection")
    
    # Initialize Polygon client if needed
    if 'POLYGON_API_KEY' not in os.environ:
        st.error("POLYGON_API_KEY not found in environment variables. Please set it in the .env file.")
        st.stop()
    
    if st.session_state.polygon_client is None:
        st.session_state.polygon_client = init_polygon_client()
    
    # Show ticker input
    ticker = st.text_input(
        'Stock Ticker', 
        'AAPL', 
        help='Enter the stock ticker symbol (e.g., AAPL, MSFT)',
        key='ticker_input'
    ).upper()
    
    # Always use real-time data
    use_sample_data = False

# Start metrics server in a separate thread
def start_monitoring():
    server = start_metrics_server(host="0.0.0.0", port=8000)
    server.run()

# Start the metrics server in a daemon thread
monitoring_thread = threading.Thread(target=start_monitoring, daemon=True)
monitoring_thread.start()

def sarima_analysis(ticker, data):
    st.title("SARIMA Time Series Analysis")
    st.write(f"Analyzing {ticker} stock prices using SARIMA model")
    
    # Prepare time series data
    ts_data = data['Close'].dropna()
    
    with st.spinner('Analyzing time series data...'):
        # Display original time series
        st.subheader("Original Time Series")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data, name='Close Price'))
        fig.update_layout(
            title='Stock Price Over Time',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Decompose the time series
        st.subheader("Time Series Decomposition")
        
        # Calculate a safe period based on data length
        # Use the largest possible period that fits in the data with at least 2 full cycles
        max_possible_period = min(252, len(ts_data) // 2)  # Max 1 year (252 trading days)
        safe_period = min(5, max_possible_period)  # Start with 5 as minimum
        
        # Try to find a reasonable period that fits the data
        for period in [252, 63, 21, 5]:  # Yearly, quarterly, monthly, weekly
            if len(ts_data) >= 2 * period:
                safe_period = period
                break
                
        st.info(f"Using adaptive period of {safe_period} days for decomposition")
        
        try:
            # Use STL for more robust decomposition with limited data
            from statsmodels.tsa.seasonal import STL
            
            try:
                # Try with the calculated period
                stl = STL(ts_data, period=safe_period, robust=True)
                decomposition = stl.fit()
            except Exception as e:
                # If that fails, try with a smaller period
                st.warning(f"Decomposition with period {safe_period} failed, trying with smaller period...")
                safe_period = max(5, safe_period // 2)
                stl = STL(ts_data, period=safe_period, robust=True)
                decomposition = stl.fit()
            
            # Store the period for reference
            st.session_state.seasonal_period = safe_period
            
            # Plot decomposition
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
            
            # Original series
            ax1.plot(ts_data.index, ts_data)
            ax1.set_title('Original Time Series')
            
            # Trend
            ax2.plot(ts_data.index, decomposition.trend)
            ax2.set_title('Trend')
            
            # Seasonal
            ax3.plot(ts_data.index, decomposition.seasonal)
            ax3.set_title('Seasonal')
            
            # Residual
            ax4.plot(ts_data.index, decomposition.resid)
            ax4.set_title('Residual')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Could not perform seasonal decomposition: {str(e)}")
        
        # ACF and PACF plots
        st.subheader("Autocorrelation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Autocorrelation Function (ACF)")
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_acf(ts_data, lags=40, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.write("Partial Autocorrelation Function (PACF)")
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_pacf(ts_data, lags=40, ax=ax)
            st.pyplot(fig)
        
        # SARIMA Model Fitting
        st.subheader("SARIMA Model Fitting")
        st.write("Automatically finding the best SARIMA parameters using pmdarima...")
        
        with st.spinner('Fitting SARIMA model (this may take a few minutes)...'):
            try:
                # Use auto_arima to find best SARIMA parameters
                model = pm.auto_arima(ts_data,
                                    start_p=0, max_p=3,
                                    start_q=0, max_q=3,
                                    d=1,  # Let the model determine differencing
                                    seasonal=True,
                                    m=5,  # Weekly seasonality (5 trading days)
                                    start_P=0, max_P=2,
                                    start_Q=0, max_Q=2,
                                    D=1,  # Seasonal differencing
                                    trace=True,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)
                
                st.success(f"Best SARIMA parameters: {model.order}x{model.seasonal_order}")
                
                # Plot diagnostics
                st.subheader("Model Diagnostics")
                fig = model.plot_diagnostics(figsize=(12, 8))
                st.pyplot(fig)
                
                # Make forecasts
                st.subheader("Forecast")
                n_periods = st.slider("Select number of days to forecast:", 5, 30, 10)
                
                # Generate forecasts
                forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
                
                # Use the actual datetime index for plotting if available
                if isinstance(ts_data.index, pd.DatetimeIndex):
                    # Create forecast index based on data type
                    last_date = ts_data.index[-1]
                    forecast_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=len(forecast),
                        freq='B'  # Business days only
                    )
                else:
                    forecast_dates = range(len(ts_data), len(ts_data) + len(forecast))
                xaxis_title = 'Date'
                
                fig = go.Figure()
                
                # Plot historical data
                fig.add_trace(go.Scatter(
                    x=historical_index[-100:],  # Last 100 periods
                    y=ts_data[-100:],
                    mode='lines',
                    name='Historical Data'
                ))
                
                # Ensure forecast_dates is the same length as forecast
                if len(forecast_dates) > len(forecast):
                    forecast_dates = forecast_dates[:len(forecast)]
                elif len(forecast_dates) < len(forecast):
                    forecast = forecast[:len(forecast_dates)]
                
                # Plot forecast
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='green')
                ))
                
                # Plot confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast_dates + forecast_dates[::-1],  # x coordinates for the polygon
                    y=np.concatenate([conf_int[:, 0], conf_int[:, 1][::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=True,
                    name='95% Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f'{n_periods}-Day Stock Price Forecast',
                    xaxis_title=xaxis_title,
                    yaxis_title='Stock Price',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast values in a table
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecast': forecast,
                    'Lower Bound': conf_int[:, 0],
                    'Upper Bound': conf_int[:, 1]
                }).set_index('Date')
                
                st.dataframe(
                    forecast_df.style.format({
                        'Forecast': '{:.2f}',
                        'Lower Bound': '{:.2f}',
                        'Upper Bound': '{:.2f}'
                    })
                )
                
                # Model summary
                st.subheader("Model Summary")
                st.text(str(model.summary()))
                
                # Add testing section
                tab1, tab2 = st.tabs(["Analysis", "Testing"])
                
                with tab1:
                    st.subheader("Model Summary")
                    st.text(str(model.summary()))
                    
                    # Show forecast plot in the Analysis tab
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(
                        forecast_df.style.format({
                            'Forecast': '{:.2f}',
                            'Lower Bound': '{:.2f}',
                            'Upper Bound': '{:.2f}'
                        })
                    )
                
                with tab2:
                    st.header("Model Testing")
                    
                    # Calculate in-sample predictions
                    try:
                        # Get in-sample predictions
                        pred = model.predict_in_sample()
                        
                        # Calculate metrics
                        mse = mean_squared_error(ts_data, pred)
                        mae = mean_absolute_error(ts_data, pred)
                        rmse = np.sqrt(mse)
                        mape = np.mean(np.abs((ts_data - pred) / ts_data)) * 100
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MSE", f"{mse:.2f}")
                        with col2:
                            st.metric("MAE", f"{mae:.2f}")
                        with col3:
                            st.metric("RMSE", f"{rmse:.2f}")
                        with col4:
                            st.metric("MAPE %", f"{mape:.2f}")
                        
                        # Plot actual vs predicted
                        fig_test = go.Figure()
                        fig_test.add_trace(go.Scatter(
                            x=ts_data.index,
                            y=ts_data,
                            mode='lines',
                            name='Actual'
                        ))
                        fig_test.add_trace(go.Scatter(
                            x=ts_data.index,
                            y=pred,
                            mode='lines',
                            name='Predicted'
                        ))
                        fig_test.update_layout(
                            title='Actual vs Predicted Values',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            template='plotly_white',
                            showlegend=True
                        )
                        st.plotly_chart(fig_test, use_container_width=True)
                        
                        # Plot residuals
                        residuals = ts_data - pred
                        fig_resid = go.Figure()
                        fig_resid.add_trace(go.Scatter(
                            x=ts_data.index,
                            y=residuals,
                            mode='lines+markers',
                            name='Residuals'
                        ))
                        fig_resid.add_hline(
                            y=0,
                            line_dash='dash',
                            line_color='red',
                            opacity=0.5
                        )
                        fig_resid.update_layout(
                            title='Residuals Plot',
                            xaxis_title='Date',
                            yaxis_title='Residuals',
                            template='plotly_white',
                            showlegend=True
                        )
                        st.plotly_chart(fig_resid, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error in testing section: {str(e)}")
                    
            except Exception as e:
                st.error(f"Error fitting SARIMA model: {str(e)}")

# Page routing and model parameters
look_back = 60  # Number of previous time steps to use for prediction
years_of_data = 5  # Number of years of historical data to use

# SARIMA parameters (different from LSTM's lookback/epochs)
sarima_order = (1, 1, 1)  # (p,d,q) - ARIMA order
seasonal_order = (1, 1, 1, 5)  # (P,D,Q,s) - seasonal order (5 for weekly seasonality)

# Main app logic will handle the page routing

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {font-size:24px; font-weight: bold; color: #1f77b4;}
    .metric-value {font-size: 20px; font-weight: bold;}
    .stProgress > div > div > div > div {background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);}
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Price Prediction")
st.markdown("""
This application uses an LSTM neural network to predict stock prices. The model is trained on historical price data
and can make predictions for future time periods. The visualization shows the actual prices versus the predicted prices.
""")

# Sidebar for user inputs
st.sidebar.title('Stock Prediction Settings')

# Data source selection - Default to sample data
use_sample_data = st.sidebar.checkbox(
    'Use Sample Data', 
    value=True,  # Default to True
    help='Using sample data to avoid API issues. Uncheck to fetch real-time data.')

# Ticker input (only show if not using sample data)
ticker = 'AAPL'  # Default ticker
if not use_sample_data:
    ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL').upper()
    st.sidebar.warning("Note: Real-time data requires a valid Polygon.io API key")

# Lookback period
look_back = st.sidebar.slider(
    'Lookback Period (days)', 
    min_value=30, 
    max_value=252,  # Reduced from 1095 to 1 year for stability
    value=60,  # Default to 60 days
    step=1,
    help='Number of previous days to use for prediction (lower values work better with sample data)')

# Years of data to use
years_of_data = st.sidebar.slider(
    'Years of Historical Data', 
    min_value=1, 
    max_value=5,  # Maximum 5 years
    value=1,  # Default to 1 year
    step=1,
    help='Number of years of historical data to use for training')

# Model training settings
st.sidebar.subheader('Model Settings')
epochs = st.sidebar.slider(
    'Training Epochs', 
    min_value=10, 
    max_value=100,  # Reduced from 200 to 100
    value=30,  # Reduced from 50 to 30 for faster training
    step=5,
    help='Number of training iterations (lower values train faster)')

batch_size = st.sidebar.selectbox(
    'Batch Size', 
    options=[16, 32, 64],  # Removed 128
    index=1,  # Default to 32
    help='Number of samples per gradient update')

# Display options
st.sidebar.subheader('Display Options')
show_train_data = st.sidebar.checkbox(
    'Show Training Data', 
    value=True,
    help='Show the training data visualization')

show_metrics = st.sidebar.checkbox(
    'Show Model Metrics', 
    value=True,
    help='Show model performance metrics')

# Add some spacing and info
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Using optimized settings for better performance and stability")

def fetch_stock_data(ticker: str, years: int = 5) -> Optional[pd.DataFrame]:
    """
    Fetch stock data using Polygon.io API with multiple features
    
    Args:
        ticker: Stock ticker symbol
        years: Number of years of historical data to fetch
        
    Returns:
        DataFrame with stock data or None if there was an error
    """
    # Use environment variable or session state client
    if st.session_state.polygon_client is None and 'POLYGON_API_KEY' in os.environ:
        st.session_state.polygon_client = init_polygon_client(os.environ['POLYGON_API_KEY'])
    
    if st.session_state.polygon_client is None:
        st.warning("Polygon.io API key not found. Using sample data instead.")
        return None
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    try:
        monitor.record_api_call('polygon', 'started')
        start_time = time.time()
        
        # Get data using pagination to ensure we get all available data
        all_aggs = []
        current_date = start_date
        
        try:
            while current_date < end_date:
                next_date = min(current_date + timedelta(days=365), end_date)
                
                # Get aggregated bars (candles) data
                aggs = list(st.session_state.polygon_client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan='day',
                    from_=current_date.strftime('%Y-%m-%d'),
                    to=next_date.strftime('%Y-%m-%d'),
                    limit=50000
                ))
                
                if aggs:
                    all_aggs.extend(aggs)
                
                current_date = next_date + timedelta(days=1)
                
        except Exception as e:
            st.warning(f"Error fetching data for {ticker} from {current_date}: {str(e)}")
            return None
        
        if not all_aggs:
            monitor.record_api_call('polygon', 'no_data')
            st.error(f"No data found for ticker: {ticker}")
            return None
        
        # Convert to DataFrame and clean up
        data = []
        for agg in all_aggs:
            try:
                data.append({
                    'Open': float(agg.open),
                    'High': float(agg.high),
                    'Low': float(agg.low),
                    'Close': float(agg.close),
                    'Volume': int(agg.volume),
                    'Date': datetime.utcfromtimestamp(agg.timestamp / 1000).strftime('%Y-%m-%d')
                })
            except (ValueError, AttributeError):
                continue
        
        if not data:
            st.error("No valid data points found in the response")
            return None
            
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.drop_duplicates('Date').sort_values('Date')
        df = df.set_index('Date')
        
        # Ensure we have data for the full date range
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        df = df.reindex(date_range).fillna(method='ffill')
        
        # Ensure we have all required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                monitor.record_api_call('polygon', 'missing_columns')
                st.warning(f"Required column {col} not found in data for {ticker}")
                return None
        
        # Record successful API call
        monitor.record_api_call('polygon', 'success')
        monitor.record_data_metrics(len(df))
        monitor.record_custom_metric('last_data_fetch_time', time.time() - start_time, 
                                   'Time taken to fetch data from Polygon.io')
                
        return df
    except Exception as e:
        monitor.record_api_call('polygon', 'error')
        st.error(f"Error fetching data from Polygon.io: {str(e)}")
        return None

def load_sample_stock_data(ticker: str = 'AAPL') -> pd.DataFrame:
    """
    Load sample stock data from Excel file based on ticker
    
    Args:
        ticker: Stock ticker symbol (currently only 'AAPL' is supported for sample data)
        
    Returns:
        DataFrame with stock data
    """
    try:
        # Currently we only have AAPL sample data
        if ticker.upper() != 'AAPL':
            st.warning(f"Sample data is only available for AAPL. Using AAPL data instead of {ticker}.")
            
        excel_file = "sample_aapl_data.xlsx"
        
        # Check if file exists
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"Sample data file not found: {excel_file}")
            
        # Try to read the Excel file
        try:
            # First read without setting index to check for required columns
            df = pd.read_excel(excel_file, engine='openpyxl')
            
            # Check if DataFrame is empty
            if df.empty:
                raise ValueError("Sample data file is empty")
                
            # Ensure we have all required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in sample data file: {', '.join(missing_cols)}")
                
            # Set Date as index and sort
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df = df.sort_index()
            
            # Keep only the required columns in the right order
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            st.info(f"Successfully loaded data from {excel_file}")
            return df
            
        except ImportError:
            st.error("openpyxl is required to read Excel files. Install it with: pip install openpyxl")
            raise
        
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        # If we get here, something went wrong with the Excel file
        # Fall back to generating sample data
        st.warning("Falling back to generated sample data")
        
        # Generate sample data
        n_points = 1000
        dates = pd.date_range(end=datetime.now(), periods=n_points, freq='B')
        base = np.linspace(100, 200, n_points)
        noise = np.random.normal(0, 5, n_points)
        
        close_prices = base + noise
        open_prices = close_prices * (1 + np.random.normal(0, 0.005, n_points))
        high_prices = close_prices * (1 + np.random.uniform(0, 0.01, n_points))
        low_prices = close_prices * (1 - np.random.uniform(0, 0.01, n_points))
        
        high_prices = np.maximum(high_prices, open_prices, close_prices)
        low_prices = np.minimum(low_prices, open_prices, close_prices)
        
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': np.random.randint(100000, 1000000, size=n_points)
        }, index=dates)
        
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

def fetch_realtime_data(ticker: str, years: int = 5) -> Optional[pd.DataFrame]:
    """
    Fetch stock data, using sample data for AAPL and Polygon API for other tickers
    
    Args:
        ticker: Stock ticker symbol
        years: Number of years of historical data to fetch (not used for sample data)
        
    Returns:
        DataFrame with stock data or None if there was an error
    """
    try:
        # For AAPL, use the sample data from Excel
        if ticker.upper() == 'AAPL':
            st.info("Using sample data for AAPL...")
            return load_sample_stock_data('AAPL')
            
        # For other tickers, use Polygon API
        if st.session_state.get('polygon_client') is None:
            raise ValueError("Polygon API client not initialized. Please check your API key and try again.")
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        st.info(f"Fetching {years} years of data for {ticker} from Polygon...")
        
        # Convert dates to strings in YYYY-MM-DD format
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        try:
            # Get the client from session state
            client = st.session_state.polygon_client
            
            # First, check if the ticker is valid by getting aggregates
            aggs = []
            for a in client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan='day',
                from_=start_str,
                to=end_str,
                limit=50000  # Max limit
            ):
                aggs.append(a)
            
            if not aggs:
                raise ValueError(f"No data returned for ticker: {ticker}. Please check the ticker symbol.")
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'Date': pd.to_datetime(agg.timestamp, unit='ms'),
                'Open': agg.open,
                'High': agg.high,
                'Low': agg.low,
                'Close': agg.close,
                'Volume': agg.volume
            } for agg in aggs])
            
            if df.empty:
                raise ValueError(f"No valid data points found for {ticker}")
                
            # Set index and sort
            df = df.set_index('Date').sort_index()
            
            # Record successful fetch
            monitor.record_api_call('polygon', 'success')
            monitor.record_data_metrics(len(df))
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error fetching data from Polygon: {str(e)}")
            
    except Exception as e:
        monitor.record_api_call('polygon', 'error')
        st.error(f"Error: {str(e)}")
        return None

def load_and_prepare_data(ticker: str, look_back: int, years_of_data: int = 1, use_sample_data: bool = False) -> Tuple[pd.DataFrame, bool]:
    """
    Load and prepare data for training with enhanced validation
    
    Args:
        ticker: Stock ticker symbol
        look_back: Number of lookback periods needed for the model
        years_of_data: Number of years of historical data to fetch (only used for non-AAPL tickers)
        use_sample_data: Kept for backward compatibility, not used
        
    Returns:
        Tuple of (data, is_real_data) where data is a DataFrame and is_real_data is a boolean
    """
    def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Helper function to prepare and validate a DataFrame"""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Invalid or empty DataFrame provided")
            
        # Ensure we have all required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
        # Ensure index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # Sort by index and filter business days
        df = df.sort_index()
        return df[df.index.dayofweek < 5]  # 0=Monday, 4=Friday

    try:
        # For AAPL, always use sample data
        if ticker.upper() == 'AAPL':
            st.info("Using sample data for AAPL...")
            data = load_sample_stock_data('AAPL')
            return prepare_dataframe(data), False
            
        # For other tickers, use real-time data
        st.info(f"Fetching real-time data for {ticker}...")
        data = fetch_realtime_data(ticker, years_of_data)
        
        if data is None or data.empty:
            raise ValueError(f"No data available for {ticker}")
            
        return prepare_dataframe(data), True
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.warning("Falling back to AAPL sample data...")
        data = load_sample_stock_data('AAPL')
        return prepare_dataframe(data), False

@st.cache_data(ttl=3600, show_spinner="Training the model...")
def train_model(ticker: str, look_back: int, data: pd.DataFrame):
    """Train the LSTM model and return the predictor, model, and predictions"""
    # Record model training start
    monitor.record_custom_metric('model_training_start', time.time())
    
    predictor = StockPredictor(ticker, data.index[0].strftime('%Y-%m-%d'), 
                             data.index[-1].strftime('%Y-%m-%d'))
    
    # Record data metrics
    monitor.record_data_metrics(len(data))
    
    # Prepare data
    predictor.prepare_data(data, look_back=look_back)
    
    # Build and train the model
    model, history = predictor.train_model(epochs=50, batch_size=32, look_back=look_back)
    
    # Record training completion
    monitor.record_custom_metric('model_training_end', time.time())
    
    # Make predictions
    train_predict_plot, test_predict_plot, predicted_prices = predictor.predict()
    
    # Record prediction metrics if we have actual values to compare with
    if hasattr(predictor, 'y_test') and predicted_prices is not None:
        y_true = predictor.scaler.inverse_transform(
            np.concatenate([
                np.zeros((len(predictor.X_test), len(predictor.feature_columns) - 1)),
                predictor.y_test.reshape(-1, 1)
            ], axis=1)
        )[:, -1]
        
        monitor.record_prediction_metrics(y_true, predicted_prices.flatten())
        
        # Calculate and record additional metrics
        mae = np.mean(np.abs(y_true - predicted_prices.flatten()))
        mse = np.mean((y_true - predicted_prices.flatten()) ** 2)
        
        monitor.record_custom_metric('prediction_mae', mae, 'Mean Absolute Error of predictions')
        monitor.record_custom_metric('prediction_mse', mse, 'Mean Squared Error of predictions')
    
    return predictor, model, history, predicted_prices

def sarima_analysis(ticker, data):
    st.title("SARIMA Time Series Analysis")
    st.write(f"Analyzing {ticker} stock prices using SARIMA model")
    
    # Prepare time series data with proper datetime index
    ts_data = data['Close'].dropna()
    
    # Ensure we have a datetime index
    if not isinstance(ts_data.index, pd.DatetimeIndex):
        ts_data.index = pd.to_datetime(ts_data.index)
    
    # Display original time series
    st.subheader("Original Time Series")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data, name='Close Price'))
    fig.update_layout(
        title='Stock Price Over Time',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Decompose the time series
    st.subheader("Time Series Decomposition")
    
    # Calculate a safe period based on data length
    min_period = 5  # Minimum period for decomposition
    max_period = min(252, len(ts_data) // 2)  # Maximum 1 year (252 trading days)
    
    # Find the best period that fits the data
    best_period = min_period
    for period in [252, 63, 21, 5]:  # Yearly, quarterly, monthly, weekly
        if len(ts_data) >= 2 * period:
            best_period = period
            break
    
    st.info(f"Using adaptive period of {best_period} days for decomposition")
    
    try:
        # Try decomposition with the best period
        decomposition = seasonal_decompose(ts_data, period=best_period)
        
        # Plot decomposition
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
        
        # Original series
        ax1.plot(ts_data.index, ts_data)
        ax1.set_title('Original Time Series')
        
        # Trend
        ax2.plot(ts_data.index, decomposition.trend)
        ax2.set_title('Trend')
        
        # Seasonal
        ax3.plot(ts_data.index, decomposition.seasonal)
        ax3.set_title('Seasonal')
        
        # Residual
        ax4.plot(ts_data.index, decomposition.resid)
        ax4.set_title('Residual')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Could not perform seasonal decomposition: {str(e)}")
    
    # ACF and PACF plots
    st.subheader("Autocorrelation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Autocorrelation Function (ACF)")
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(ts_data, lags=40, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.write("Partial Autocorrelation Function (PACF)")
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_pacf(ts_data, lags=40, ax=ax)
        st.pyplot(fig)
    
    # SARIMA Model Fitting
    st.subheader("SARIMA Model Fitting")
    st.write("Automatically finding the best SARIMA parameters using pmdarima...")
    
    with st.spinner('Fitting SARIMA model (this may take a few minutes)...'):
        try:
            # Use auto_arima to find best SARIMA parameters
            model = pm.auto_arima(ts_data,
                                start_p=0, max_p=3,
                                start_q=0, max_q=3,
                                d=None,  # Let the model determine differencing
                                seasonal=True,
                                m=5,  # Weekly seasonality (5 trading days)
                                start_P=0, max_P=2,
                                start_Q=0, max_Q=2,
                                D=None,  # Let the model determine seasonal differencing
                                trace=True,
                                error_action='warn',  # Show warnings
                                suppress_warnings=False,
                                stepwise=True,
                                n_jobs=-1)  # Use all available cores
            
            st.success(f"Best SARIMA parameters: {model.order}x{model.seasonal_order}")
            
            # Plot diagnostics
            st.subheader("Model Diagnostics")
            fig = model.plot_diagnostics(figsize=(12, 8))
            st.pyplot(fig)
            
            # Make forecasts
            st.subheader("Forecast")
            n_periods = st.slider("Select number of days to forecast:", 5, 30, 10)
            
            # Generate forecasts
            forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
            
            # Create a consistent x-axis for both historical and forecast data
            if isinstance(ts_data.index, pd.DatetimeIndex):
                # If we have datetime index, use proper business days for forecast
                last_date = ts_data.index[-1]
                forecast_dates = pd.bdate_range(
                    start=last_date + pd.offsets.BDay(1),  # Next business day
                    periods=len(forecast),
                    freq='B'  # Business days only
                )
                
                # Combine historical and forecast dates for x-axis
                all_dates = ts_data.index.union(forecast_dates)
                xaxis_title = 'Date'
                
                # Create a numeric index for plotting to avoid datetime issues
                x_axis = range(len(ts_data) + len(forecast))
                
                # Create a mapping from date to x-value
                date_to_x = {date: i for i, date in enumerate(all_dates)}
                
                # Get x-values for historical data
                hist_x = [date_to_x[date] for date in ts_data.index[-100:]]
                
                # Get x-values for forecast
                forecast_x = [date_to_x[date] for date in forecast_dates]
                
                # Create custom tick labels
                tick_vals = list(range(0, len(all_dates), max(1, len(all_dates)//10)))
                tick_text = [all_dates[i].strftime('%Y-%m-%d') for i in tick_vals]
                
            else:
                # Fallback to simple numeric index if no datetime index
                forecast_x = list(range(len(ts_data), len(ts_data) + len(forecast)))
                hist_x = list(range(len(ts_data) - 100, len(ts_data)))
                xaxis_title = 'Time Period (Relative)'
                tick_vals = None
                tick_text = None
            
            fig = go.Figure()
            
            # Plot historical data with proper x-values
            fig.add_trace(go.Scatter(
                x=hist_x,
                y=ts_data[-100:].values,
                mode='lines',
                name='Historical Data'
            ))
            
            # Plot forecast
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='green')
            ))
            
            # Plot confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast_x + forecast_x[::-1],  # x coordinates for the polygon
                y=np.concatenate([conf_int[:, 0], conf_int[:, 1][::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='95% Confidence Interval'
            ))
            
            # Configure x-axis
            if tick_vals is not None:
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    tickangle=45
                )
            
            fig.update_layout(
                title=f'{n_periods}-Period Stock Price Forecast',
                xaxis_title=xaxis_title,
                yaxis_title='Stock Price',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show forecast values in a table with proper dates if available
            if isinstance(ts_data.index, pd.DatetimeIndex):
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecast': forecast,
                    'Lower Bound': conf_int[:, 0],
                    'Upper Bound': conf_int[:, 1]
                }).set_index('Date')
            else:
                forecast_df = pd.DataFrame({
                    'Day': [f'Day {i+1}' for i in range(len(forecast))],
                    'Forecast': forecast,
                    'Lower Bound': conf_int[:, 0],
                    'Upper Bound': conf_int[:, 1]
                }).set_index('Day')
            
            st.dataframe(
                forecast_df.style.format({
                    'Forecast': '{:.2f}',
                    'Lower Bound': '{:.2f}',
                    'Upper Bound': '{:.2f}'
                })
            )
            
            # Model summary
            st.subheader("Model Summary")
            st.text(str(model.summary()))
            
            # Add testing section
            st.markdown("---")
            st.header("Model Performance")
            
            # Calculate in-sample predictions
            try:
                # Get in-sample predictions
                pred = model.predict_in_sample()
                
                # Calculate metrics
                mse = mean_squared_error(ts_data, pred)
                mae = mean_absolute_error(ts_data, pred)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((ts_data - pred) / ts_data)) * 100
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MSE", f"{mse:.2f}")
                with col2:
                    st.metric("MAE", f"{mae:.2f}")
                with col3:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col4:
                    st.metric("MAPE", f"{mape:.2f}%")
                
                # Plot actual vs predicted
                fig = go.Figure()
                
                # Use the same date handling as in the forecast section
                if isinstance(ts_data.index, pd.DatetimeIndex):
                    x_vals = ts_data.index
                    x_title = 'Date'
                else:
                    x_vals = range(len(ts_data))
                    x_title = 'Time Period'
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=ts_data,
                    mode='lines',
                    name='Actual'
                ))
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=pred,
                    mode='lines',
                    name='Predicted'
                ))
                fig.update_layout(
                    title='Actual vs Predicted Values',
                    xaxis_title=x_title,
                    yaxis_title='Price',
                    template='plotly_white',
                    xaxis=dict(
                        tickangle=45,
                        showgrid=True
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in model testing: {str(e)}")
                st.exception(e)  # Show full traceback for debugging
            
        except Exception as e:
            st.error(f"Error fitting SARIMA model: {str(e)}")

# Main app logic
if 'train_clicked' not in st.session_state:
    st.session_state.train_clicked = False

# Load data for the selected page
try:
    if page == "LSTM Prediction":
        data, is_real_data = load_and_prepare_data(
            ticker=ticker.upper(),
            look_back=look_back,
            years_of_data=years_of_data,
            use_sample_data=False
        )
        if data is None or data.empty:
            st.error("Failed to load data. Please try a different ticker or use sample data.")
            st.stop()
    elif page == "SARIMA Analysis":
        data = fetch_realtime_data(ticker.upper())
        if data is None or data.empty:
            st.error("Failed to load data for SARIMA analysis. Please try a different ticker.")
            st.stop()
        sarima_analysis(ticker, data)
        st.stop()
        
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

if st.sidebar.button('Train Model') or st.session_state.train_clicked:
    st.session_state.train_clicked = True
    with st.spinner('Training model...'):
        # Train model with the loaded data
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Add a callback for training progress
        class TrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / 50  # 50 is the default number of epochs
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f'Epoch {epoch + 1}/50 - loss: {logs["loss"]:.4f} - val_loss: {logs["val_loss"]:.4f}')
        
        # Train the model with progress tracking
        predictor, model, history, predicted_prices = train_model(
            ticker, look_back, data
        )
        
        # Clean up progress bar
        progress_bar.empty()
        status_text.empty()
        
        # Get actual prices for the test period with the same length as predictions
        actual_prices = predictor.test_data.reshape(-1, 1)[:len(predicted_prices)]
        
        # Generate dates for the test period up to today
        end_date = datetime.now()
        start_date = end_date - timedelta(days=len(actual_prices) - 1)
        test_dates = pd.date_range(
            start=start_date,
            end=end_date,
            periods=len(actual_prices)
        )
        
        # Generate future dates for prediction starting from tomorrow
        future_days = 60
        future_dates = pd.date_range(
            start=end_date + timedelta(days=1),
            periods=future_days
        )
        
        # Prepare the last sequence for prediction
        last_sequence = predictor.X_test[-1:]
        future_predictions = []
        
        # Generate predictions for future days
        future_predictions = []
        current_sequence = np.copy(predictor.X_test[-1:])
        
        for _ in range(future_days):
            # Predict next day
            next_pred_scaled = predictor.model.predict(current_sequence, verbose=0)[0][0]
            
            # Get the last sequence and update it
            new_sequence = np.roll(current_sequence[0], -1, axis=0)
            
            # Create a dummy row with the predicted close price
            dummy_row = np.zeros(len(predictor.feature_columns))
            dummy_row[predictor.feature_columns.index('Close')] = next_pred_scaled
            
            # Update the last position with the new prediction
            new_sequence[-1] = dummy_row
            
            # Store the prediction (in original scale)
            # We need to inverse transform just the close price
            dummy_features = np.zeros((1, len(predictor.feature_columns)))
            dummy_features[0, predictor.feature_columns.index('Close')] = next_pred_scaled
            next_pred_original = predictor.scaler.inverse_transform(dummy_features)[0, predictor.feature_columns.index('Close')]
            future_predictions.append(next_pred_original)
            
            # Update the sequence for next prediction
            current_sequence = np.array([new_sequence])
        
        future_predictions_original = np.array(future_predictions)
        
        # Create DataFrames for visualization
        df_past = pd.DataFrame({
            'Date': test_dates,
            'Type': 'Actual',
            'Price': actual_prices.flatten()
        })
        
        df_predicted = pd.DataFrame({
            'Date': test_dates,
            'Type': 'Predicted',
            'Price': predicted_prices.flatten()
        })
        
        df_future = pd.DataFrame({
            'Date': future_dates,
            'Type': 'Future Prediction',
            'Price': future_predictions_original
        })
        
        # Combine all data
        df = pd.concat([df_past, df_predicted, df_future])
        
        # Calculate metrics
        mse = np.mean((actual_prices - predicted_prices) ** 2)
        mae = np.mean(np.abs(actual_prices - predicted_prices))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        
        # Display metrics if enabled
        if show_metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Squared Error", f"{mse:.2f}")
            with col2:
                st.metric("Mean Absolute Error", f"{mae:.2f}")
            with col3:
                st.metric("Root Mean Squared Error", f"{rmse:.2f}")
            with col4:
                st.metric("Mean Absolute % Error", f"{mape:.2f}%")
        
        # Create the main plot
        fig = go.Figure()
        
        # Calculate date ranges for plotting
        total_days = len(predictor.train_data) + len(actual_prices)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=total_days)
        all_dates = pd.date_range(start=start_date, end=end_date, periods=total_days)
        train_dates = all_dates[:len(predictor.train_data)]
        test_dates = all_dates[-len(actual_prices):]
        
        # Add training data if selected
        if show_train_data:
            fig.add_trace(go.Scatter(
                x=train_dates,
                y=predictor.train_data.flatten(),
                mode='lines',
                name='Training Data',
                line=dict(color='#1f77b4', width=2)
            ))
        
        # Add actual and predicted prices
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=actual_prices.flatten(),
            mode='lines',
            name='Actual Prices',
            line=dict(color='#2ca02c', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=predicted_prices.flatten(),
            mode='lines',
            name='Predicted Prices',
            line=dict(color='#ff7f0e', width=3, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='Stock Price Prediction',
            xaxis_title='Date',
            yaxis_title='Stock Price',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_white',
            hovermode='x unified',
            height=600
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a separate graph for future predictions
        st.subheader('Future Price Forecast')
        future_df = df[df['Type'] == 'Future Prediction']
        
        # Create a new figure for future predictions
        fig_future = go.Figure()
        
        # Add the last actual price point for reference
        last_actual_date = df[df['Type'] == 'Actual']['Date'].max()
        last_actual_price = df[(df['Date'] == last_actual_date) & (df['Type'] == 'Actual')]['Price'].values[0]
        
        # Add a marker for the last actual price
        fig_future.add_trace(go.Scatter(
            x=[last_actual_date],
            y=[last_actual_price],
            mode='markers',
            name='Last Actual Price',
            marker=dict(color='blue', size=10)
        ))
        
        # Add the future predictions
        fig_future.add_trace(go.Scatter(
            x=future_df['Date'],
            y=future_df['Price'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='green', width=2)
        ))
        
        # Add a line connecting the last actual to first prediction
        if len(future_df) > 0:
            fig_future.add_trace(go.Scatter(
                x=[last_actual_date, future_df['Date'].iloc[0]],
                y=[last_actual_price, future_df['Price'].iloc[0]],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ))
        
        # Update layout for future predictions graph
        fig_future.update_layout(
            title=f'Next {future_days} Days Price Forecast',
            xaxis_title='Date',
            yaxis_title='Predicted Price',
            template='plotly_white',
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        # Display the future predictions graph
        st.plotly_chart(fig_future, use_container_width=True)
        
        # Show the predictions in a table below the graph
        st.subheader('Detailed Predictions')
        st.dataframe(
            future_df[['Date', 'Price']]
            .rename(columns={'Price': 'Predicted Price'})
            .set_index('Date')
            .style.format({'Predicted Price': '${:,.2f}'})
            .background_gradient(cmap='Greens', subset=['Predicted Price'])
        )

# Add some space at the bottom
st.markdown("---")
st.markdown("### About")
st.markdown("""
This application demonstrates the use of LSTM neural networks for time series forecasting of stock prices.
The model is trained on historical price data and can make predictions for future time periods.

**Note:** This is a demonstration using sample data. For real-world applications, consider using
more sophisticated models and additional features.
""")

if __name__ == "__main__":
    # Let Streamlit handle the server configuration
    import streamlit.web.cli as st_cli
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--deploy":
        st_cli._main_run_clExplicit(
            "app.py",
            "streamlit run",
            []
        )
