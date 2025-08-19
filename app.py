import os
import time
import threading
import uvicorn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from stock_prediction import StockPredictor, load_sample_data
from typing import Tuple, Optional, Dict, Any
from polygon import RESTClient
from dotenv import load_dotenv

# Import monitoring
from monitoring import monitor
from metrics_server import start_metrics_server

# Load environment variables
load_dotenv()

# Initialize Polygon client
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    st.error("Polygon API key not found. Please check your .env file.")
    st.stop()

polygon_client = RESTClient(api_key=POLYGON_API_KEY)

# Start metrics server in a separate thread
def start_monitoring():
    server = start_metrics_server(host="0.0.0.0", port=8000)
    server.run()

# Start the metrics server in a daemon thread
monitoring_thread = threading.Thread(target=start_monitoring, daemon=True)
monitoring_thread.start()

# Set page config
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {font-size:24px; font-weight: bold; color: #1f77b4;}
    .metric-value {font-size: 20px; font-weight: bold;}
    .stProgress > div > div > div > div {background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);}
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Price Prediction with LSTM")
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

# Years of data to use (only show if not using sample data)
years_of_data = 5  # Default value
if not use_sample_data:
    years_of_data = st.sidebar.slider(
        'Years of Historical Data', 
        min_value=1, 
        max_value=5,  # Reduced from 10 to 5 for stability
        value=3,  # Default to 3 years
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
    # Check if Polygon API key is available
    if 'POLYGON_API_KEY' not in os.environ or not os.environ['POLYGON_API_KEY']:
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
        
        while current_date < end_date:
            # Get one year of data at a time to avoid timeouts
            batch_end = min(current_date + timedelta(days=365), end_date)
            
            try:
                batch_aggs = list(polygon_client.list_aggs(
                    ticker=ticker.upper(),
                    multiplier=1,
                    timespan="day",
                    from_=current_date.strftime('%Y-%m-%d'),
                    to=batch_end.strftime('%Y-%m-%d'),
                    limit=50000,
                    sort='asc'
                ))
                
                if batch_aggs:
                    all_aggs.extend(batch_aggs)
                
            except Exception as e:
                st.warning(f"Error fetching data for {ticker} from {current_date} to {batch_end}: {str(e)}")
                
            current_date = batch_end + timedelta(days=1)
        
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

def load_sample_stock_data() -> Tuple[pd.DataFrame, bool]:
    """Generate sample stock data for demonstration"""
    # Generate sample data
    np.random.seed(42)
    n_points = 1000
    dates = pd.date_range(end=datetime(2025, 8, 4), periods=n_points)
    base = np.linspace(100, 200, n_points)
    noise = np.random.normal(0, 5, n_points)
    
    # Create realistic stock-like data
    close_prices = base + noise
    open_prices = close_prices * (1 + np.random.normal(0, 0.005, n_points))
    high_prices = close_prices * (1 + np.random.uniform(0, 0.01, n_points))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.01, n_points))
    
    # Ensure high > open,close > low
    high_prices = np.maximum(high_prices, open_prices, close_prices)
    low_prices = np.minimum(low_prices, open_prices, close_prices)
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': np.random.randint(100000, 1000000, size=n_points)
    }, index=dates)
    
    return df, False

def load_and_prepare_data(ticker: str, look_back: int, years_of_data: int = 5, use_sample_data: bool = False) -> Tuple[pd.DataFrame, bool]:
    """
    Load and prepare data for training with enhanced validation
    
    Args:
        ticker: Stock ticker symbol
        look_back: Number of lookback periods needed for the model
        years_of_data: Number of years of historical data to fetch
        use_sample_data: Whether to use sample data instead of real-time data
        
    Returns:
        Tuple of (data, is_real_data) where data is a DataFrame and is_real_data is a boolean
    """
    try:
        if use_sample_data:
            st.info("Using sample data for demonstration")
            data, _ = load_sample_stock_data()
            is_real_data = False
        else:
            # Calculate how many years of data we need
            # We want at least 2x lookback periods for train/test split
            min_days_needed = look_back * 2
            years_needed = max(1, int(np.ceil(min_days_needed / 252)) + 1)  # Add buffer
            years_needed = min(years_needed, 10)  # Cap at 10 years
            
            st.info(f"Fetching up to {years_needed} years of data for {ticker}...")
            data = fetch_stock_data(ticker, years=years_needed)
            
            if data is None:
                st.warning(f"Could not fetch data for {ticker}. Using sample data instead.")
                data, _ = load_sample_stock_data()
                is_real_data = False
            else:
                is_real_data = True
        
        # Ensure we have enough data after loading
        if len(data) < look_back * 2:
            st.warning(f"Insufficient data points. Need at least {look_back * 2}, but only have {len(data)}.")
            st.info("Using sample data instead.")
            data, _ = load_sample_stock_data()
            is_real_data = False
        
        # Ensure we have all required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            st.warning(f"Missing required columns in data: {missing_cols}")
            st.info("Using sample data instead.")
            data, _ = load_sample_stock_data()
            is_real_data = False
            
        # Ensure data is properly sorted
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
            
        # Ensure we have business days only
        data = data[data.index.dayofweek < 5]  # 0=Monday, 4=Friday
        
        return data, is_real_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Falling back to sample data.")
        data, _ = load_sample_stock_data()
        return data, False

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

# Main app logic
if 'train_clicked' not in st.session_state:
    st.session_state.train_clicked = False

# Load data first (not cached)
data, is_real_data = load_and_prepare_data(ticker, look_back, years_of_data, use_sample_data)

if st.sidebar.button('Train Model') or st.session_state.train_clicked:
    st.session_state.train_clicked = True
    with st.spinner('Training model...'):
        # Train model with the loaded data
        predictor, model, history, predicted_prices = train_model(
            ticker, look_back, data
        )
        
        # Get actual prices for the test period with the same length as predictions
        actual_prices = predictor.test_data.reshape(-1, 1)[:len(predicted_prices)]
        
        # Generate dates for the test period only
        test_dates = pd.date_range(
            end=datetime(2025, 8, 4),
            periods=len(actual_prices)
        )
        
        # Generate future dates for prediction
        last_date = test_dates[-1]
        future_days = 60
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
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
