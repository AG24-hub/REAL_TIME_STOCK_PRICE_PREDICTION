**📈 Stock Price Prediction Dashboard <br>**
A machine learning-powered web application that predicts future stock prices using deep learning (LSTM Neural Network) and displays live market indices like NIFTY 50 and SENSEX. Built with Streamlit, TensorFlow, yfinance, and Matplotlib.<br><br>

**🚀 Features<br>**<br>
✅ Predict future stock prices for the next 60 days <br>
✅ Choose from major Indian stocks: RELIANCE, TCS, INFY, ICICI BANK, SBIN <br>
✅ Visualize: <br>
    - Closing price trends over 10 years <br>
    - 1-year zoom view <br>
    - 100 & 200-day moving averages <br>
✅ Show live NIFTY 50 and SENSEX data <br>
✅ Download predicted prices as CSV <br>
✅ Smooth UX with spinner and caching for fast prediction <br><br>

**🧠 Tech Stack** <br>
Python for 	Core programming <br>
Streamlit for Interactive dashboard frontend <br>
TensorFlow Keras for Deep learning LSTM model <br>
joblib for	Model and scaler persistence <br>
yfinance for Real-time & historical stock data <br>
Matplotlib for Data visualizations <br>
Pandas / NumPy	for  Data manipulation & math <br>
scikit-learn  for 	Scaling and preprocessing <br> <br>

**🗂 Project Structure**
📁 saved_models/  <br>
   ├── stock_model.h5         # Trained LSTM model  <br>
   └── scaler.pkl             # Saved MinMaxScaler  <br>
📄 app.py                     # Streamlit app file <br>
📄 requirements.txt           # Dependencies <br>
📄 README.md                  # You're here! <br><br>

**⚙️ How It Works**
- App loads a saved LSTM model trained on 10 years of stock data
- Takes the last 60 days as input
- Predicts the next N days based on previous trends
- Displays future price chart and allows CSV export

**📦 Installation**
1. Clone the repo:
  git clone https://github.com/AG24-hub/REAL_TIME_STOCK_PRICE_PREDICTION.git
  cd REAL_TIME_STOCK_PRICE_PREDICTION
2. Install dependencies:
   pip install -r requirements.txt
3. streamlit run app.py

🌐 Try it Online 

