import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import io

st.set_page_config(page_title="Finance ML App", layout="wide")
st.markdown("<h1 style='color:#4B8BBE;'>📊 Finance Machine Learning App</h1>", unsafe_allow_html=True)
st.image("https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif", width=300)


st.sidebar.header("🔍 Load Data")
uploaded_file = st.sidebar.file_uploader("Upload Kragle Dataset (CSV)", type=["csv"])
stock_symbol = st.sidebar.text_input("Enter Yahoo Finance Ticker (e.g., AAPL, UBL)", value="AAPL")

df = None
yahoo_df = None


if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Kragle dataset uploaded successfully!")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# fetching data from yahoo finance
if st.sidebar.button("Fetch Yahoo Data"):
    try:
        yahoo_df = yf.download(stock_symbol, period="6mo", interval="1d")
        df = yahoo_df.reset_index()
        st.success("✅ Yahoo Finance data fetched!")
        st.write(df.head())
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")


if st.button("🔧 Preprocess Data"):
    if df is not None:
        st.info("Preprocessing: Handling missing values...")
        st.write("Missing values before:", df.isnull().sum())
        df.dropna(inplace=True)
        st.success("✅ Data cleaned. Missing values removed.")
        st.write("Shape after cleaning:", df.shape)
    else:
        st.warning("❗ Please upload or fetch data first.")

# feauture engineering 
if st.button("🧠 Feature Engineering"):
    if df is not None:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            st.warning("⚠️ Not enough numeric columns for ML.")
        else:
            features = numeric_df.columns.tolist()
            st.write("Numeric Features:", features)
            st.session_state['features'] = features
            st.session_state['data'] = numeric_df
            st.success("✅ Feature selection complete.")
    else:
        st.warning("❗ Please upload or fetch data first.")

#traing and splitting the dataa
if st.button("✂️ Train/Test Split"):
    if 'data' in st.session_state:
        X = st.session_state['data'].iloc[:, :-1]
        y = st.session_state['data'].iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.session_state.update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        })

        fig, ax = plt.subplots()
        ax.pie([len(X_train), len(X_test)], labels=["Train", "Test"], autopct="%1.1f%%")
        st.pyplot(fig)
        st.success("✅ Data split complete.")
    else:
        st.warning("❗ Please complete feature engineering first.")


if st.button("🏋️ Train Model"):
    if all(key in st.session_state for key in ['X_train', 'y_train']):
        model = LinearRegression()
        model.fit(st.session_state['X_train'], st.session_state['y_train'])
        st.session_state['model'] = model
        st.success("✅ Linear Regression model trained.")
    else:
        st.warning("❗ Please split the data first.")


if st.button("📏 Evaluate Model"):
    if 'model' in st.session_state:
        y_pred = st.session_state['model'].predict(st.session_state['X_test'])
        mse = mean_squared_error(st.session_state['y_test'], y_pred)
        r2 = r2_score(st.session_state['y_test'], y_pred)

        st.metric("📉 Mean Squared Error", f"{mse:.2f}")
        st.metric("📈 R² Score", f"{r2:.2f}")

        st.session_state['y_pred'] = y_pred
    else:
        st.warning("❗ Train the model first.")

if st.button("📊 Show Predictions"):
    if 'y_pred' in st.session_state:
        pred_df = pd.DataFrame({
            "Actual": st.session_state['y_test'].values,
            "Predicted": st.session_state['y_pred']
        })
        fig = px.line(pred_df, title="Actual vs Predicted")
        st.plotly_chart(fig)
        st.session_state['results'] = pred_df
    else:
        st.warning("❗ Evaluate the model first.")


if st.button("📥 Download Results"):
    if 'results' in st.session_state:
        buffer = io.StringIO()
        st.session_state['results'].to_csv(buffer, index=False)
        st.download_button("Download CSV", data=buffer.getvalue(), file_name="predictions.csv", mime="text/csv")
    else:
        st.warning("❗ No results to download yet.")


st.title("📈 Stock Price Visualization")

stock_symbol_input = st.text_input("Enter Stock Symbol", "AAPL")

if stock_symbol_input:
    stock_data = yf.download(stock_symbol_input, start="2018-01-01", end="2023-01-01")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_data['Close'], label=f"{stock_symbol_input} Close Price")
    ax.set_title(f"{stock_symbol_input} Stock Price")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

