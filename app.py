import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ---------------------------
# Load data and train model (cached)
# ---------------------------
@st.cache_resource
def load_and_train():
    # Ice Cream.csv must be in the same folder as app.py
    df = pd.read_csv("Ice Cream.csv")

    X = df[["Temperature"]].values
    y = df[["Revenue"]].values

    # You can also keep a train/test split if you want, but for prediction
    # it is fine to train on all data
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    return model, df

model, df = load_and_train()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Ice Cream Revenue Predictor")
st.write("Predict ice‑cream revenue from temperature using a Decision Tree Regressor.")

st.subheader("Dataset preview")
st.dataframe(df.head())

temp = st.number_input(
    "Temperature (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=30.0,
    step=0.5,
)

if st.button("Predict revenue"):
    X_new = np.array([[temp]])            # shape (1, 1)
    y_pred = model.predict(X_new)

    revenue = float(np.ravel(y_pred)[0])  # handle possible (1,1) output

    st.success(f"Predicted revenue: {revenue:.2f}")
