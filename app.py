import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

@st.cache_resource
def load_and_train():
    # CSV must be in the same folder as app.py
    df = pd.read_csv("Ice Cream.csv")

    X = df[["Temperature"]].values
    y = df[["Revenue"]].values

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    return model, df

model, df = load_and_train()
