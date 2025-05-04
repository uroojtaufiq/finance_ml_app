import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Finance ML App", layout="wide")

# CSS styling 
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #4B8BBE, #F8D210);
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: white;
            text-align: center;
            font-size: 3em;
            margin-top: 20px;
        }
        .stButton>button {
            background-color: #4B8BBE;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #FF5733;
            color: white;
            transform: scale(1.05);
        }
        .stTextInput>label {
            font-weight: bold;
        }
        .stFileUploader>label {
            font-weight: bold;
        }
        .stSidebar {
            background-color: #f1f1f1;
            transition: all 0.3s ease;
        }
        .stSidebar:hover {
            background-color: #e2e2e2;
        }
        footer {
            text-align: center;
            font-size: 0.8em;
            color: #4B8BBE;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1>📊 Finance Machine Learning App</h1>", unsafe_allow_html=True)
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

# Fetch Yahoo Finance Data
if st.sidebar.button("Fetch Yahoo Data"):
    try:
        # Fetch Yahoo Finance data and save to CSV
        yahoo_df = yf.download(stock_symbol, period="6mo", interval="1d")
        yahoo_df.to_csv(f"{stock_symbol}_data.csv")
        df = yahoo_df.reset_index()
        st.success(f"✅ {stock_symbol} data fetched and saved as CSV!")
        st.write(df.head())
        st.download_button("Download CSV", data=yahoo_df.to_csv(), file_name=f"{stock_symbol}_data.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")

# user can select model from here
model_choice = st.selectbox("Choose Model", ["Linear Regression", "Decision Tree", "Random Forest"])

# Preprocessng step
with st.expander("🔧 Preprocess Data", expanded=True):
    if st.button("Preprocess Data"):
        if df is not None:
            st.info("Cleaning dataset: Converting to numeric, filling missing values...")

            st.write("Missing values before cleaning:", df.isnull().sum())

            # Convert everything possible to numeric to avoid errorsss
            df = df.apply(pd.to_numeric, errors='coerce')

            # Drop columns with more than 20% missing values
            df.dropna(axis=1, thresh=len(df) * 0.8, inplace=True)

            # Fill remaining missing values using forward fill
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)  

            st.success("✅ Data cleaned. Missing values filled, unnecessary columns dropped.")
            st.write("Shape after cleaning:", df.shape)
            st.session_state['df'] = df  
        else:
            st.warning("❗ Please upload or fetch data first.")

# Handling Missing Data in this step
missing_option = st.selectbox("How would you like to handle missing values?", ["Forward Fill", "Backward Fill", "Drop", "Interpolate"])
if missing_option == "Forward Fill":
    df.fillna(method='ffill', inplace=True)
elif missing_option == "Backward Fill":
    df.fillna(method='bfill', inplace=True)
elif missing_option == "Drop":
    df.dropna(inplace=True)
elif missing_option == "Interpolate":
    df.interpolate(method='linear', inplace=True)

# Feature Engineering
with st.expander("🧠 Feature Engineering", expanded=True):
    if st.button("Feature Engineering"):
        if 'df' in st.session_state:
            df = st.session_state['df']
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
            st.warning("❗ Please upload or fetch and preprocess data first.")

# Trainng and testing the data 
with st.expander("✂️ Train/Test Split", expanded=True):
    if st.button("Train/Test Split"):
        if 'data' in st.session_state:
            X = st.session_state['data'].iloc[:, :-1]
            y = st.session_state['data'].iloc[:, -1]

          
            if len(X) < 5:
                st.warning("❗ Not enough data points after preprocessing for ML. Need at least 5 rows.")
            else:
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

# Model Training 
with st.expander("📏 Evaluate Model", expanded=True):
    if st.button("Train Model"):
        if 'X_train' in st.session_state and 'y_train' in st.session_state:
            # Choose the selected model
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Decision Tree":
                model = DecisionTreeRegressor()
            elif model_choice == "Random Forest":
                model = RandomForestRegressor()

            model.fit(st.session_state['X_train'], st.session_state['y_train'])

          
            y_pred = model.predict(st.session_state['X_test'])

            mse = mean_squared_error(st.session_state['y_test'], y_pred)
            r2 = r2_score(st.session_state['y_test'], y_pred)

            st.write(f"Mean Squared Error: {mse}")
            st.write(f"R-Squared: {r2}")

            # Feature Importance Visualization (for Random Forest or Decision Tree)
            if model_choice in ["Random Forest", "Decision Tree"]:
                feature_importance = model.feature_importances_
                st.bar_chart(feature_importance)

     
            st.session_state['y_pred'] = y_pred
            st.session_state['model'] = model
            st.success("✅ Model trained and evaluated.")

# Show Predictions
with st.expander("📊 Show Predictions", expanded=True):
    if 'y_pred' in st.session_state:
        st.write("Predictions vs Actual Values:")
        predictions_df = pd.DataFrame({
            'Actual': st.session_state['y_test'],
            'Predicted': st.session_state['y_pred']
        })
        st.write(predictions_df)
        
        # Plot predictions vs actual
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(st.session_state['y_test'], st.session_state['y_pred'])
        ax.plot([min(st.session_state['y_test']), max(st.session_state['y_test'])],
                [min(st.session_state['y_test']), max(st.session_state['y_test'])], 'r--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        st.pyplot(fig)
    else:
        st.warning("❗ Train the model first to view predictions.")

# Download from here
with st.expander("📥 Download Results", expanded=True):
    if 'y_pred' in st.session_state:
        result_df = pd.DataFrame({
            'Actual': st.session_state['y_test'],
            'Predicted': st.session_state['y_pred']
        })
        result_csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=result_csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    else:
        st.warning("❗ Train the model first to download results.")


st.markdown("""
    <footer>
        <p>Finance ML App | Created by Your Name</p>
        <p>Data Source: Yahoo Finance | Libraries: Streamlit, scikit-learn, yfinance</p>
    </footer>
""", unsafe_allow_html=True)



