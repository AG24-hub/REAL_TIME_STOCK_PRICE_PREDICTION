📈 Stock Price Prediction Dashboard <br>
A machine learning-powered web application that predicts future stock prices using deep learning (LSTM Neural Network) and displays live market indices like NIFTY 50 and SENSEX. Built with Streamlit, TensorFlow, yfinance, and Matplotlib.

🚀 Features
✅ Predict future stock prices for the next 60 days
✅ Choose from major Indian stocks: RELIANCE, TCS, INFY, ICICI BANK, SBIN
✅ Visualize:
    - Closing price trends over 10 years
    - 1-year zoom view
    - 100 & 200-day moving averages
✅ Show live NIFTY 50 and SENSEX data
✅ Download predicted prices as CSV
✅ Smooth UX with spinner and caching for fast prediction

🧠 Tech Stack
Tool / Library	Purpose
Python	Core programming
Streamlit	Interactive dashboard frontend
TensorFlow / Keras	Deep learning LSTM model
joblib	Model and scaler persistence
yfinance	Real-time & historical stock data
Matplotlib	Data visualizations
Pandas / NumPy	Data manipulation & math
scikit-learn	Scaling and preprocessing

🗂 Project Structure
bash
Copy
Edit
📁 saved_models/
   ├── stock_model.h5         # Trained LSTM model
   └── scaler.pkl             # Saved MinMaxScaler

📄 app.py                     # Streamlit app file
📄 requirements.txt           # Dependencies
📄 README.md                  # You're here!
⚙️ How It Works
App loads a saved LSTM model trained on 10 years of stock data

Takes the last 60 days as input

Predicts the next N days based on previous trends

Displays future price chart and allows CSV export

📦 Installation
1. Clone the repo:
  git clone https://github.com/your-username/stock-price-predictor.git
  cd stock-price-predictor
2. Install dependencies:
   pip install -r requirements.txt
3. streamlit run app.py

🌐 Try it Online 

