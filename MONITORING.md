# ML Model Monitoring with Grafana and Prometheus

This document explains how to set up and use the monitoring system for the LSTM Stock Predictor.

## Architecture

The monitoring system consists of:

1. **Prometheus** - Time-series database for storing metrics
2. **Grafana** - Visualization platform for creating dashboards
3. **FastAPI Metrics Server** - Exposes model metrics in Prometheus format
4. **Python Prometheus Client** - Instrumentation library for collecting metrics

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Monitoring Stack

Use Docker Compose to start Prometheus and Grafana:

```bash
docker-compose up -d
```

### 3. Start the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The metrics server will start automatically on port 8000.

## Accessing Dashboards

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin` (change this in production)

- **Prometheus**: http://localhost:9090

## Available Metrics

### Model Metrics
- `model_train_loss` - Training loss
- `model_validation_loss` - Validation loss
- `model_train_mae` - Training MAE
- `model_validation_mae` - Validation MAE
- `model_prediction_errors` - Histogram of prediction errors

### API Metrics
- `api_calls_total` - Count of API calls by endpoint and status
- `data_points_total` - Number of data points processed

### Custom Metrics
- `prediction_mae` - Mean Absolute Error of predictions
- `prediction_mse` - Mean Squared Error of predictions
- `last_data_fetch_time` - Time taken to fetch data from Polygon.io
- `model_training_start` - Timestamp when model training started
- `model_training_end` - Timestamp when model training completed

## Grafana Dashboard

A pre-configured dashboard is available at:
- **ML Model Monitoring Dashboard** - Shows training metrics, prediction errors, and API statistics

## Customizing Metrics

To add custom metrics:

1. Import the monitor in your Python code:
   ```python
   from monitoring import monitor
   ```

2. Record metrics using the monitor:
   ```python
   # Record a gauge metric
   monitor.record_custom_metric('metric_name', value, 'Description of the metric')
   
   # Record a counter
   monitor.record_api_call('endpoint_name', 'status')
   ```

## Troubleshooting

1. **No data in Grafana**
   - Verify Prometheus is scraping the metrics endpoint (http://localhost:8000/metrics)
   - Check the Prometheus targets page (http://localhost:9090/targets)

2. **Connection refused**
   - Ensure the metrics server is running (starts automatically with the app)
   - Check if port 8000 is available

3. **Missing metrics**
   - Verify the model has been trained at least once
   - Check the application logs for errors
