import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# ---------------------------
# 1. Create data & train model (cached)
# ---------------------------
@st.cache_resource
def load_and_train():
    # Synthetic dataset: Temperature vs Revenue
    # You can change this to match your own pattern.
    np.random.seed(42)
    temperatures = np.linspace(0, 40, 80)  # 0°C to 40°C
    # Revenue roughly linear + some noise
    revenues = 200 + 50 * temperatures + np.random.normal(0, 150, size=temperatures.shape)

    df = pd.DataFrame({
        "Temperature": temperatures,
        "Revenue": revenues
    })

    X = df[["Temperature"]].values
    y = df[["Revenue"]].values

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    return model, df


model, df = load_and_train()

# ---------------------------
# 2. Streamlit UI
# ---------------------------
st.title("Ice Cream Revenue Predictor (Decision Tree Regression)")
st.write("This app predicts ice‑cream revenue from temperature using a Decision Tree Regressor trained on a synthetic dataset (no CSV file).")

st.subheader("Sample of the training data")
st.dataframe(df.head())

st.subheader("Make a prediction")

temp = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=50.0,
    value=30.0,
    step=0.5,
)

if st.button("Predict revenue"):
    X_new = np.array([[temp]])          # shape (1, 1)
    y_pred = model.predict(X_new)
    revenue = float(np.ravel(y_pred)[0])

    st.success(f"Predicted revenue: {revenue:.2f}")

# (Optional) show the full dataset as a chart
st.subheader("Training data scatter plot")
st.scatter_chart(df.rename(columns={"Temperature": "x", "Revenue": "y"}))

