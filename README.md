# 🧠 StockPredictorPro — Your Friendly Stock LSTM + Streamlit App

Hi buddy! 👋 We will make a smart app that looks at past stock prices and guesses the future. It also shows news, crypto, and more. Follow the steps slowly. You got this! 💪

## 📁 Folder Setup (We made it for you already)
```
StockPredictorPro/
├─ app/
│  ├─ streamlit_app.py
│  └─ data/            # favorites + history will be saved here
├─ notebooks/
│  └─ 01_lstm_stock_prediction.ipynb
├─ models/             # trained models will be saved here
├─ assets/             # charts/images you save will be here
├─ requirements.txt
└─ README.md
```

## 🐍 1. Make a Python place (Conda is easiest)
Open **Anaconda Prompt** and run:
```
cd path	o\StockPredictorPro
conda create -n stockpro python=3.10 -y
conda activate stockpro
pip install -r requirements.txt
```

## 📓 2. Run the Jupyter Notebook
```
jupyter notebook
```
Then open **notebooks/01_lstm_stock_prediction.ipynb** and run cells top to bottom.  
It will train the LSTM and save the model into **models/** automatically.

## 🖥️ 3. Run the Streamlit App
In the same activated environment:
```
streamlit run app/streamlit_app.py
```
- The app refreshes **every minute** to get fresh data.
- Pick a stock (AAPL, GOOGL, TSLA, MSFT, AMZN, META, NVDA, NFLX, JPM, WMT).
- Switch **Themes** in the sidebar 🎨.
- Click **⭐ Add to Favorites** to save your best stocks.
- See **News + Sentiment**, **Crypto**, and **Economic Indicators**.
- Click **🔮 Quick Predict (LSTM)** to train a small model and save it in **models/**.
- Your **history** of predictions is saved in **app/data/history.csv**.

## 🛟 Tips
- If minute data isn't available, the app falls back to safe intervals.
- This is **educational**, not financial advice.
- If something breaks, read the error message and try again calmly 🙂

Happy building! 🚀
