import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
import plotly.graph_objects as go
import streamlit as st
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Any

# Try to import yfinance with a fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    if 'streamlit' in globals():
        st.warning("yfinance is not installed. Some features may be limited.")
    else:
        print("Warning: yfinance is not installed. Some features may be limited.")

# Import monitoring
from monitoring import monitor

class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            monitor.record_training_metrics(self.model.history)

class StockPredictor:
    def __init__(self, ticker: str, start_date: str = None, end_date: str = None):
        self.ticker = ticker
        self.start_date = start_date or (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.train_data = None
        self.test_data = None
        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.target_column = 'Close'

    def fetch_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch stock data using yfinance with multiple features
        
        Returns:
            DataFrame with stock data or None if data couldn't be fetched
        """
        if not YFINANCE_AVAILABLE:
            if 'streamlit' in globals():
                st.warning("yfinance is not available. Cannot fetch live data.")
            else:
                print("Warning: yfinance is not available. Cannot fetch live data.")
            return None
            
        try:
            df = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            if df.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            # Ensure we have all required columns
            for col in self.feature_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column {col} not found in data")
                    
            return df[self.feature_columns]
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def prepare_data(self, data: pd.DataFrame, look_back: int = 60, train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare the data for training and testing
        
        Args:
            data: DataFrame containing the stock data
            look_back: Number of previous time steps to use for prediction
            train_size: Fraction of data to use for training (0.0 to 1.0)
            
        Returns:
            Tuple of (X, y) arrays for training
            
        Raises:
            ValueError: If there's insufficient data or invalid parameters
        """
        # Input validation
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
            
        if look_back < 1:
            raise ValueError("look_back must be at least 1")
            
        if not (0 < train_size < 1):
            raise ValueError("train_size must be between 0 and 1")
            
        # Ensure required columns exist
        required_columns = set(self.feature_columns + [self.target_column])
        if not required_columns.issubset(data.columns):
            missing = required_columns - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}")
            
        # Ensure we have enough data
        min_required = max(look_back * 2, 100)  # At least 2x lookback or 100 points
        if len(data) < min_required:
            raise ValueError(
                f"Insufficient data points. Need at least {min_required} points, "
                f"but only have {len(data)}. Try reducing the lookback period or getting more data."
            )
            
        # Handle missing values
        if data.isnull().any().any():
            print("Data contains missing values. Filling with forward then backward fill.")
            data = data.ffill().bfill()
            
            # If still missing values, drop those rows
            if data.isnull().any().any():
                initial_count = len(data)
                data = data.dropna()
                dropped = initial_count - len(data)
                print(f"Dropped {dropped} rows with missing values after filling.")
                
                if len(data) < min_required:
                    raise ValueError(
                        f"After handling missing values, only {len(data)} points remain, "
                        f"but need at least {min_required}."
                    )
        
        # Scale the features
        self.scaler = MinMaxScaler()
        scaled_features = self.scaler.fit_transform(data[self.feature_columns])
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(scaled_features) - look_back):
            X.append(scaled_features[i:(i + look_back)])
            y.append(scaled_features[i + look_back, self.feature_columns.index(self.target_column)])
            
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets
        train_size = int(len(X) * train_size)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]
        
        # Store the original data for plotting
        self.train_data = data[self.target_column].values[look_back:train_size + look_back]
        self.test_data = data[self.target_column].values[train_size + look_back:len(data)]
        
        print(f"Data split into {len(self.X_train)} training samples and {len(self.X_test)} test samples.")
        
        return X, y

    def build_model(self, look_back: int = 252) -> None:
        """Build an enhanced LSTM model with better architecture for long-term forecasting."""
        self.model = tf.keras.Sequential([
            # First LSTM layer with return sequences
            tf.keras.layers.LSTM(
                units=128,
                return_sequences=True,
                input_shape=(look_back, len(self.feature_columns)),
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Second LSTM layer
            tf.keras.layers.LSTM(
                units=64,
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Third LSTM layer
            tf.keras.layers.LSTM(
                units=32,
                return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Dense layers for final prediction
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=1)
        ])
        
        # Use learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )
        return self.model

    def train_model(self, epochs=100, batch_size=32, look_back=252):
        """Train the LSTM model and return both model and history"""
        if self.model is None:
            self.build_model(look_back=look_back)
        
        # Create a temporary directory for model checkpoints
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, 'best_model.h5')
        
        try:
            # Create model checkpoint to save best model
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,  # Only save weights to avoid model serialization issues
                mode='min',
                verbose=1
            )
            
            # Early stopping to prevent overfitting
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
            
            # Learning rate reducer
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
            
            # Add metrics callback
            metrics_callback = MetricsCallback()
            
            # Training with validation split
            history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[
                    early_stopping,
                    model_checkpoint,
                    reduce_lr,
                    metrics_callback
                ],
                verbose=1,
                shuffle=False  # Important for time series data
            )
            
            # Load the best weights if they exist
            if os.path.exists(model_path):
                self.model.load_weights(model_path)
                
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory: {e}")
        
        # Record final metrics
        monitor.record_training_metrics(history)
        
        return self.model, history

    def predict(self, X: np.ndarray = None, look_back: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions using a rolling window approach, where each prediction
        is based on previous predictions, making the forecast more realistic.
        
        Args:
            X: Input data (defaults to test data if None)
            look_back: Number of time steps used in the model
            
        Returns:
            Tuple of (train_predict_plot, test_predict_plot, predicted_prices)
        """
        if X is None:
            X = self.X_test
            
        # Get the scaler's min and scale for the target column
        target_idx = self.feature_columns.index(self.target_column)
        
        print(f"Input shape: {X.shape}")
        
        # Get the last window from training data to start predictions
        last_window = self.X_train[-1:]
        
        # Initialize array to store predictions
        predicted_sequences = []
        
        # Make predictions one step at a time, updating the input window each time
        for i in range(len(X)):
            # Predict next value
            current_pred = self.model.predict(last_window, verbose=0)
            
            # Store the prediction
            predicted_sequences.append(current_pred[0, 0])
            
            # Update the last window with the new prediction
            if i < len(X) - 1:
                # Create a new window by shifting the old one and adding the prediction
                new_window = np.roll(last_window, -1, axis=1)
                new_window[0, -1, target_idx] = current_pred[0, 0]
                last_window = new_window
        
        # Convert predictions to numpy array
        predictions = np.array(predicted_sequences).reshape(-1, 1)
        
        # Inverse transform the predictions to get actual prices
        predicted_prices = []
        for i in range(len(predictions)):
            # Create a dummy array with the same shape as the input features
            dummy_array = np.zeros((1, len(self.feature_columns)))
            dummy_array[0, target_idx] = predictions[i, 0]
            predicted_price = self.scaler.inverse_transform(dummy_array)[0, target_idx]
            predicted_prices.append(predicted_price)
            
        predicted_prices = np.array(predicted_prices)
        
        # Get the training data for plotting
        if hasattr(self.train_data, 'columns') and self.target_column in self.train_data.columns:
            train_target = self.train_data[self.target_column].values.reshape(-1, 1)
        elif isinstance(self.train_data, np.ndarray):
            # If train_data is a numpy array, assume the target is the first column
            train_target = self.train_data.reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported train_data type: {type(self.train_data)}")
        
        # Create arrays for plotting
        total_length = len(train_target) + len(predicted_prices)
        train_predict_plot = np.empty((total_length, 1))
        train_predict_plot[:, :] = np.nan
        train_predict_plot[:len(train_target), :] = train_target
        
        test_predict_plot = np.empty((total_length, 1))
        test_predict_plot[:, :] = np.nan
        
        # Fill in the test predictions
        if len(predicted_prices) > 0:
            test_start = len(train_target)
            test_end = test_start + len(predicted_prices)
            test_predict_plot[test_start:test_end, :] = predicted_prices.reshape(-1, 1)
        
        return train_predict_plot, test_predict_plot, predicted_prices.reshape(-1, 1)

    def evaluate(self, predicted_prices):
        """Evaluate model performance"""
        # Get the actual prices for the test period
        actual_prices = self.test_data[:len(predicted_prices)]
        
        # Ensure predicted_prices is 1D
        if len(predicted_prices.shape) > 1:
            predicted_prices = predicted_prices.flatten()
        
        # Ensure actual_prices is 1D
        if len(actual_prices.shape) > 1:
            actual_prices = actual_prices.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(actual_prices, predicted_prices)
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mse)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((actual_prices - predicted_prices) / (actual_prices + 1e-10))) * 100  # Add small constant to avoid division by zero
        
        print("\nModel Evaluation:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        # Print sample predictions vs actuals
        print("\nSample predictions vs actual:")
        for i in range(min(5, len(actual_prices))):
            # Safely access values regardless of array dimensions
            pred = predicted_prices[i] if isinstance(predicted_prices[i], (int, float)) else predicted_prices[i][0]
            actual = actual_prices[i] if isinstance(actual_prices[i], (int, float)) else actual_prices[i][0]
            print(f"Day {i+1}: Predicted = {pred:.2f}, Actual = {actual:.2f}")
        
        return mse, mae, rmse

def load_sample_data():
    """Load sample data from Excel file"""
    try:
        print("Loading AAPL sample data from Excel...")
        df = pd.read_excel('sample_aapl_data.xlsx')
        
        # Ensure we have all required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Sample data is missing required columns")
            
        print(f"Loaded {len(df)} days of AAPL data.")
        print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        return df
        
    except Exception as e:
        print(f"Error loading sample data: {e}")
        print("Falling back to generated sample data...")
        import generate_sample_data
        generate_sample_data.generate_sample_data()
        df = pd.read_csv('sample_stock_data.csv')
        return df

def main():
    # Configuration
    LOOK_BACK = 60  # Number of previous days to use for prediction
    
    # Initialize the predictor with dummy values (not used for sample data)
    predictor = StockPredictor('SAMPLE', None, None)
    
    # Load sample data
    print("\nStep 1: Loading data...")
    data = load_sample_data()
    
    print("\nStep 2: Preparing data for LSTM...")
    predictor.prepare_data(data, look_back=LOOK_BACK)
    
    # Build and train the model
    predictor.build_model(look_back=LOOK_BACK)
    print("Training model...")
    history = predictor.train_model(epochs=50, batch_size=32, look_back=LOOK_BACK)
    
    # Make predictions
    print("Making predictions...")
    train_predict_plot, test_predict_plot, predicted_prices = predictor.predict()
    
    # Evaluate the model
    print("\nModel Evaluation:")
    mse, mae, rmse = predictor.evaluate(predicted_prices)
    
    # Plot the results with better visualization
    plt.figure(figsize=(16, 8))
    plt.title('Stock Price Prediction with LSTM')
    
    # Get the actual price data
    all_data = predictor.scaler.inverse_transform(
        np.concatenate((predictor.X_train.reshape(-1, len(predictor.feature_columns)), 
                       predictor.X_test.reshape(-1, len(predictor.feature_columns))), axis=0)
    )
    close_prices = all_data[:, predictor.feature_columns.index('Close')]
    
    # Define the test start point (end of training data)
    test_start = len(predictor.X_train)
    
    # Plot training data
    plt.plot(np.arange(test_start), 
             close_prices[:test_start], 
             label='Training Data',
             color='blue',
             alpha=0.7)
    
    # Plot test data
    test_len = len(predictor.X_test)
    plt.plot(np.arange(test_start, test_start + test_len),
             close_prices[test_start:test_start + test_len],
             label='Actual Test Data',
             color='green',
             alpha=0.7)
    
    # Plot predictions
    pred_start = test_start
    pred_end = pred_start + len(predicted_prices)
    plt.plot(np.arange(pred_start, pred_end),
             predicted_prices.flatten(),
             label='Predicted Prices',
             color='red',
             linewidth=2)
    
    # Add vertical line to separate training and test data
    plt.axvline(x=test_start, color='black', linestyle='--', alpha=0.7)
    
    # Add text annotation
    plt.text(test_start, 
             min(plt.ylim()[0], min(predicted_prices)), 
             'Test Data Start',
             rotation=90,
             verticalalignment='bottom')
    
    # Add prediction metrics to the plot
    plt.figtext(0.15, 0.85, 
                f'Model Performance:\nMSE: {mse:.2f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}',
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    
    plt.xlabel('Time (days)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot with higher DPI
    plt.savefig('stock_prediction.png', dpi=300, bbox_inches='tight')
    print("\nPrediction plot saved as 'stock_prediction.png'")
    plt.show()

if __name__ == "__main__":
    main()
