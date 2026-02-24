import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px

st.title("📊 Sales Forecasting & Business Intelligence Dashboard")

# =====================================================
# ✅ DIRECT DATASET LOAD (NO UPLOAD NEEDED)
# Put 'Sample - Superstore.csv' in SAME folder as app.py
# =====================================================

df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")
df.columns = df.columns.str.strip()

# Rename columns
df = df.rename(columns={
    "Order Date": "Date",
    "Sales": "Revenue",
    "Quantity": "Units_Sold"
})

df["Promotion"] = 0
df["Store"] = 1

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# ================= FEATURE ENGINEERING =================
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek

df["Lag_1"] = df["Revenue"].shift(1)
df["Rolling_Mean"] = df["Revenue"].rolling(7).mean()

df = df.dropna()

# 🔥 Keep only numeric columns (fix XGBoost error)
numeric_df = df.select_dtypes(include=[np.number])

target = "Revenue"
features = [c for c in numeric_df.columns if c != target]

X = numeric_df[features]
y = numeric_df[target]

split = int(len(numeric_df)*0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ================= MODEL =================
model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)
model.fit(X_train, y_train)

preds = model.predict(X_test)

# ================= METRICS =================
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

st.subheader("📈 Model Performance")
c1,c2,c3 = st.columns(3)
c1.metric("RMSE", round(rmse,2))
c2.metric("MAE", round(mae,2))
c3.metric("R2 Score", round(r2,2))

# ================= GRAPH =================
results = pd.DataFrame({
    "Date": df["Date"].iloc[split:],
    "Actual": y_test.values,
    "Predicted": preds
})

fig = px.line(results, x="Date", y=["Actual","Predicted"],
              title="Forecast vs Actual Sales")
st.plotly_chart(fig, use_container_width=True)

monthly = df.groupby("Month")["Revenue"].sum().reset_index()
fig2 = px.bar(monthly, x="Month", y="Revenue",
              title="Monthly Revenue")
st.plotly_chart(fig2, use_container_width=True)
