from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary, REGISTRY
import time
from typing import Dict, Any
import numpy as np

class ModelMonitor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        
        # Check if metrics already exist before creating them
        if 'model_train_loss' not in REGISTRY._names_to_collectors:
            self.train_loss = Gauge('model_train_loss', 'Training loss')
            self.val_loss = Gauge('model_validation_loss', 'Validation loss')
            self.train_mae = Gauge('model_train_mae', 'Training MAE')
            self.val_mae = Gauge('model_validation_mae', 'Validation MAE')
            
            # Metrics for predictions
            self.prediction_errors = Histogram(
                'model_prediction_errors', 
                'Prediction errors',
                buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
            )
            
            # API and data metrics
            self.api_calls = Counter('api_calls_total', 'Total API calls', ['endpoint', 'status'])
            self.data_points = Gauge('data_points_total', 'Total data points processed')
            
            # Model performance metrics
            self.model_metrics = {}
    
    def record_training_metrics(self, history):
        """Record training metrics from model history"""
        if not hasattr(self, 'train_loss'):
            return
            
        if 'loss' in history.history:
            self.train_loss.set(history.history['loss'][-1])
        if 'val_loss' in history.history:
            self.val_loss.set(history.history['val_loss'][-1])
        if 'mae' in history.history:
            self.train_mae.set(history.history['mae'][-1])
        if 'val_mae' in history.history:
            self.val_mae.set(history.history['val_mae'][-1])
    
    def record_prediction_metrics(self, y_true, y_pred):
        """Record prediction accuracy metrics"""
        if not hasattr(self, 'prediction_errors'):
            return
            
        errors = np.abs(y_true - y_pred)
        for error in errors:
            self.prediction_errors.observe(float(error))
    
    def record_data_metrics(self, data_size: int):
        """Record data-related metrics"""
        if hasattr(self, 'data_points'):
            self.data_points.set(data_size)
    
    def record_api_call(self, endpoint: str, status: str = 'success'):
        """Record API call metrics"""
        if hasattr(self, 'api_calls'):
            self.api_calls.labels(endpoint=endpoint, status=status).inc()
    
    def record_custom_metric(self, name: str, value: float, description: str = ''):
        """Record a custom metric"""
        if not hasattr(self, 'model_metrics'):
            return
            
        if name not in self.model_metrics:
            self.model_metrics[name] = Gauge(f'model_{name}', description or f'Model metric: {name}')
        self.model_metrics[name].set(value)

# Global monitor instance
monitor = ModelMonitor()
