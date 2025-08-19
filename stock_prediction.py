import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any

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
        """Fetch stock data using yfinance with multiple features"""
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

    def predict(self, X: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions and inverse transform to original scale"""
        if X is None:
            X = self.X_test
            
        # Make predictions
        predictions = self.model.predict(X)
        
        # Create a dummy array with the same shape as the original data
        # and inverse transform the predictions
        dummy_array = np.zeros((len(predictions), len(self.feature_columns)))
        # Put predictions in the 'Close' column position
        dummy_array[:, self.feature_columns.index(self.target_column)] = predictions.reshape(-1)
        
        # Inverse transform the predictions
        predicted_prices = self.scaler.inverse_transform(dummy_array)[:, self.feature_columns.index(self.target_column)]
        
        # Create arrays for plotting
        total_length = len(self.train_data) + len(predicted_prices)
        train_predict_plot = np.empty((total_length, 1))
        train_predict_plot[:, :] = np.nan
        
        test_predict_plot = np.empty((total_length, 1))
        test_predict_plot[:, :] = np.nan
        
        # Fill in the test predictions
        if len(predicted_prices) > 0:
            test_predict_plot[len(self.train_data):len(self.train_data) + len(predicted_prices)] = predicted_prices.reshape(-1, 1)
        
        return train_predict_plot, test_predict_plot, predicted_prices.reshape(-1, 1)

    def evaluate(self, predicted_prices):
        """Evaluate model performance"""
        # Get the actual prices for the test period
        actual_prices = self.test_data[:len(predicted_prices)]
        
        # Calculate metrics
        mse = mean_squared_error(actual_prices, predicted_prices)
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mse)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        
        print("\nModel Evaluation:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        # Print sample predictions vs actuals
        print("\nSample predictions vs actual:")
        for i in range(min(5, len(actual_prices))):
            print(f"Day {i+1}: Predicted = {predicted_prices[i][0]:.2f}, Actual = {actual_prices[i][0]:.2f}")
        
        return mse, mae, rmse

def load_sample_data():
    """Load sample data from CSV file"""
    import os
    if not os.path.exists('sample_stock_data.csv'):
        print("Sample data file not found. Generating sample data...")
        import generate_sample_data
        generate_sample_data.generate_sample_data()
    
    print("Loading sample stock data...")
    df = pd.read_csv('sample_stock_data.csv')
    print(f"Loaded {len(df)} days of data.")
    print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    return df['Close'].values.reshape(-1, 1)

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
    predictor.build_model()
    print("Training model...")
    history = predictor.train_model(epochs=50, batch_size=32)
    
    # Make predictions
    print("Making predictions...")
    train_predict_plot, test_predict_plot, predicted_prices = predictor.predict()
    
    # Evaluate the model
    print("\nModel Evaluation:")
    mse, mae, rmse = predictor.evaluate(predicted_prices)
    
    # Plot the results
    plt.figure(figsize=(16, 8))
    plt.title('Stock Price Prediction')
    
    # Plot original data
    plt.plot(np.arange(len(data)), data, label='Original Price')
    
    # Plot test predictions
    test_x = np.arange(len(self.train_data), len(self.train_data) + len(predicted_prices))
    plt.plot(test_x, predicted_prices, label='Predicted Price', color='red')
    
    plt.xlabel('Time (days)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('stock_prediction.png')
    print("\nPrediction plot saved as 'stock_prediction.png'")
    plt.show()

if __name__ == "__main__":
    main()
