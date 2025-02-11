import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#title 
st.title("Using Linear Regression for advertising Sales Prediction")

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Advertising.csv") 
    return data

df = load_data()
st.write("### First Five Rows of Dataset:", df.head())

# Data Preprocessing
st.write("### Data Preprocessing:")
st.write("Missing Values:", df.isnull().sum())

# Categorical Features (TV, Radio, Newspaper) and Target (Sales)
X = df[['TV', 'Radio', 'Newspaper']]  # Independent Variables
y = df['Sales']  # Dependent Variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write(f"Model Evaluation Metrics:")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"R²: {r2:.2f}")

# Visualization
st.write("Scatter Plot of TV Advertising vs Sales")
fig, ax = plt.subplots()
ax.scatter(X_test['TV'], y_test, label="Actual Sales", color="blue")
ax.scatter(X_test['TV'], y_pred, label="Predicted Sales", color="red")
ax.set_xlabel("TV Advertising Budget")
ax.set_ylabel("Sales")
ax.legend()
st.pyplot(fig)
