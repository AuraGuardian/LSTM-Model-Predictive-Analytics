# Stock Price Prediction with LSTM

A Streamlit web application that uses LSTM neural networks to predict stock prices.

## Features
- Predict stock prices using LSTM
- Uses sample data by default (no API key required)
- Option to use real-time data with Polygon.io API
- Model performance metrics and visualizations

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-prediction-lstm.git
cd stock-prediction-lstm
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For real-time data:
   - Get a free API key from [Polygon.io](https://polygon.io/)
   - Copy `.env.example` to `.env`
   - Add your API key to `.env`

5. Run the app:
```bash
streamlit run app.py
```

## Usage
1. The app runs with sample data by default
2. To use real data:
   - Uncheck "Use Sample Data" in the sidebar
   - Enter a valid stock ticker (e.g., AAPL, MSFT, GOOGL)
3. Adjust model settings as needed in the sidebar
4. Click "Train Model" to start prediction

## Note
- The `.env` file is in `.gitignore` to protect your API key
- Sample data is provided for demonstration purposes
