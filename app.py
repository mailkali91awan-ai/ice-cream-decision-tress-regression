import streamlit as st
import numpy as np
import pickle
from pathlib import Path

# ---------------------------
# Load trained model
# ---------------------------
@st.cache_resource
def load_model():
    pkl_path = Path("icecream_dt_regressor.pkl")  # must be in same folder as app.py

    if not pkl_path.exists():
        st.error(
            f"Model file '{pkl_path}' not found.\n\n"
            "Make sure icecream_dt_regressor.pkl is in the same directory as app.py."
        )
        st.stop()

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Ice Cream Sales Predictor")
st.write("Predict ice‑cream revenue from temperature using a trained Decision Tree Regressor.")

temp = st.number_input(
    "Temperature (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=30.0,
    step=0.5,
)

if st.button("Predict revenue"):
    # Model expects 2D array: [[temperature]]
    X_new = np.array([[temp]])
    y_pred = model.predict(X_new)

    # y was 2D during training (N,1), so prediction may be shape (1,1)
    revenue = float(np.ravel(y_pred)[0])

    st.success(f"Predicted revenue: {revenue:.2f}")