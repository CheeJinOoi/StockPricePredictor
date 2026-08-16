import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# Load Data
# =========================

stock = pd.read_csv(
    "data/AAPL_clean.csv",
    index_col="Date",
    parse_dates=True
)

# =========================
# Feature Engineering
# =========================

# Moving averages
stock["MA5"] = stock["Close"].rolling(window=5).mean()
stock["MA20"] = stock["Close"].rolling(window=20).mean()
stock["MA50"] = stock["Close"].rolling(window=50).mean()

# Daily return
stock["Return"] = stock["Close"].pct_change()

# Volatility
stock["Volatility20"] = (
    stock["Return"]
    .rolling(window=20)
    .std()
)

# =========================
# Lag Features
# =========================

stock["Lag1"] = stock["Close"].shift(1)
stock["Lag2"] = stock["Close"].shift(2)
stock["Lag3"] = stock["Close"].shift(3)
stock["Lag5"] = stock["Close"].shift(5)
stock["Lag10"] = stock["Close"].shift(10)

# Lagged returns
stock["ReturnLag1"] = stock["Return"].shift(1)
stock["ReturnLag2"] = stock["Return"].shift(2)
stock["ReturnLag3"] = stock["Return"].shift(3)

# =========================
# Target
# =========================

stock["Target"] = stock["Close"].shift(-1)

# Remove missing values
stock = stock.dropna()

# =========================
# Features
# =========================

features = [
    "Close",
    "MA5",
    "MA20",
    "MA50",
    "Return",
    "Volatility20",
    "Volume",
    "Lag1",
    "Lag2",
    "Lag3",
    "Lag5",
    "Lag10",
    "ReturnLag1",
    "ReturnLag2",
    "ReturnLag3"
]

X = stock[features]
y = stock["Target"]

# =========================
# Train/Test Split
# =========================

split_index = int(len(stock) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# =========================
# Baseline
# =========================

baseline_predictions = X_test["Close"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = mean_squared_error(
    y_test,
    baseline_predictions
) ** 0.5

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)

# =========================
# Linear Regression
# =========================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = mean_squared_error(
    y_test,
    linear_predictions
) ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

# =========================
# Random Forest
# =========================

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_test
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = mean_squared_error(
    y_test,
    rf_predictions
) ** 0.5

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

# =========================
# Model Comparison
# =========================

comparison = pd.DataFrame({
    "Model": [
        "Baseline",
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        baseline_mae,
        linear_mae,
        rf_mae
    ],
    "RMSE": [
        baseline_rmse,
        linear_rmse,
        rf_rmse
    ],
    "R2": [
        baseline_r2,
        linear_r2,
        rf_r2
    ]
})

print("\nModel Comparison")
print("================")
print(comparison.to_string(index=False))

# =========================
# Feature Importance
# =========================

importance = pd.Series(
    rf_model.feature_importances_,
    index=features
)

importance = importance.sort_values(
    ascending=False
)

print("\nFeature Importance")
print("==================")
print(importance)

# =========================
# Plot Predictions
# =========================

plt.figure(figsize=(14, 6))

plt.plot(
    y_test.index,
    y_test,
    label="Actual"
)

plt.plot(
    y_test.index,
    linear_predictions,
    label="Linear Regression"
)

plt.plot(
    y_test.index,
    rf_predictions,
    label="Random Forest"
)

plt.title("AAPL Actual vs Predicted Prices")
plt.xlabel("Date")
plt.ylabel("Price ($)")

plt.legend()
plt.grid(True)

plt.show()

# =========================
# Plot Feature Importance
# =========================

importance.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")

plt.grid(True)
plt.show()