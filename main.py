import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  , RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
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
# RSI
# =========================

delta = stock["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()
rs = average_gain / average_loss
stock["RSI"] = 100 - (100 / (1 + rs))

# =========================
# MACD
# =========================

ema12 = stock["Close"].ewm(span=12, adjust=False).mean()
ema26 = stock["Close"].ewm(span=26, adjust=False).mean()
stock["MACD"] = ema12 - ema26
stock["MACD_Signal"] = (stock["MACD"].ewm(span=9, adjust=False).mean())
stock["MACD_Hist"] = (stock["MACD"] - stock["MACD_Signal"])

# =========================
# Bollinger Bands
# =========================

bb_middle = stock["Close"].rolling(window=20).mean()
bb_std = stock["Close"].rolling(window=20).std()
stock["BB_Upper"] = (bb_middle +2 *bb_std)
stock["BB_Lower"] = (bb_middle -2 *bb_std)
stock["BB_Position"] = ((stock["Close"] - stock["BB_Lower"]) / (stock["BB_Upper"] - stock["BB_Lower"]))

# =========================
# Target
# =========================

stock["Target"] = stock["Close"].shift(-1)
stock["Direction"] = (stock["Close"].shift(-1) > stock["Close"]).astype(int)

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
    "ReturnLag3",

    "RSI",

    "MACD",
    "MACD_Signal",
    "MACD_Hist",

    "BB_Upper",
    "BB_Lower",
    "BB_Position"
]

X = stock[features]
y = stock["Target"]
y_direction = stock["Direction"]

# =========================
# Train/Test Split
# =========================

split_index = int(len(stock) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

y_direction_train = y_direction.iloc[:split_index]
y_direction_test = y_direction.iloc[split_index:]

# =========================
# Feature Scaling
# =========================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Scaled Linear Regression
# =========================

scaled_linear_model = LinearRegression()

scaled_linear_model.fit(
    X_train_scaled,
    y_train
)

scaled_linear_predictions = scaled_linear_model.predict(
    X_test_scaled
)

scaled_linear_mae = mean_absolute_error(
    y_test,
    scaled_linear_predictions
)

scaled_linear_rmse = mean_squared_error(
    y_test,
    scaled_linear_predictions
) ** 0.5

scaled_linear_r2 = r2_score(
    y_test,
    scaled_linear_predictions
)

print("\nScaled Linear Regression")
print("---------------------------")
print("MAE: ", scaled_linear_mae)
print("RMSE: ", scaled_linear_rmse)
print("R2: ", scaled_linear_r2)

# =========================
# Time-Series Validation
# =========================

tscv = TimeSeriesSplit(n_splits=5)

rf_mae_scores = []
rf_rmse_scores = []
rf_r2_scores =[]

print("\nRandom Forest Time-Series Validation")
print("========================================")

print("\n Time-Series Validation")
print("============================")
for fold, (train_index, validation_index) in enumerate (tscv.split(X_train), start=1):
    X_fold_train = X_train.iloc[train_index]
    X_fold_validation = X_train.iloc[validation_index]
    y_fold_train = y_train.iloc[train_index]
    y_fold_validation = y_train.iloc[validation_index]

    fold_scaler = StandardScaler()
    X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
    X_fold_validation_scaled = fold_scaler.transform(X_fold_validation)

    rf_fold_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf_fold_model.fit(X_fold_train, y_fold_train)
    fold_predictions = rf_fold_model.predict(X_fold_validation)
    fold_mae = mean_absolute_error(y_fold_validation, fold_predictions)
    fold_rmse = mean_squared_error(y_fold_validation, fold_predictions) **0.5
    fold_r2 = r2_score( y_fold_validation, fold_predictions)

    rf_mae_scores.append(fold_mae)
    rf_rmse_scores.append(fold_rmse)
    rf_r2_scores.append(fold_r2)

    baseline_fold_predictions = X_fold_validation["Close"]
    baseline_fold_mae = mean_absolute_error(y_fold_validation, baseline_fold_predictions)
    print(f"Fold {fold}: "f"RF MAE={fold_mae:.4f}, "f"Baseline MAE={baseline_fold_mae:.4f}")
    print(f"Fold{fold}:" f"MAE={fold_mae:.4f}," f"RMSE={fold_rmse:.4f}," f"R2={fold_r2:.4f}")

# =========================
# Random Forest Classifier
# =========================

direction_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

direction_model.fit(X_train, y_direction_train)
direction_predictions = direction_model.predict(X_test)
direction_accuracy = accuracy_score(y_direction_test, direction_predictions)
direction_baseline = (y_direction_train.mean() >= 0.5)
majority_direction = (y_direction_train.mode()[0])
baseline_direction_predictions = [majority_direction] * len(y_direction_test)
baseline_direction_accuracy = accuracy_score(y_direction_test, baseline_direction_predictions)
cm = confusion_matrix(y_direction_test, direction_predictions)
print("\nDirection Prediction")
print("====================")
print(f"Directional Accuracy: "f"{direction_accuracy * 100:.2f}%")
print(f"Direction Baseline Accuracy: "f"{baseline_direction_accuracy * 100:.2f}%")
print("\nConfusion Matrix")
print(cm)
print(classification_report(y_direction_test, direction_predictions, target_names=["DOWN","UP"]))

# =========================
# Grid-Search
# =========================

param_grid = {
    "n_estimators":[100,200,300],
    "max_depth" :[8,12,16],
    "min_samples_leaf":[1,2,4],
    "max_features":["sqrt", 1.0]
}

tscv = TimeSeriesSplit(n_splits =5)

rf_base = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

grid_search =GridSearchCV(
    estimator = rf_base, # the model we want to use
    param_grid = param_grid, #try all combinations specify
    cv = tscv, #User our time-series validation strategy
    scoring="neg_mean_absolute_error", #low negative more better
    n_jobs=-1,
    verbose=1 # display the level of message for running
)

grid_search.fit(
    X_train,
    y_train
)

print("\nBest Parameters:")
print(grid_search.best_params_)

best_cv_mae = -grid_search.best_score_
print("\nBest Cross-Validation Score:")
print(best_cv_mae)

best_rf_model = grid_search.best_estimator_
tuned_predictions = best_rf_model.predict(X_test)
tuned_mae = mean_absolute_error(y_test, tuned_predictions)
tuned_rmse = mean_squared_error(y_test, tuned_predictions) **0.5
tuned_r2 = r2_score(y_test, tuned_predictions)
print("\nTuned Random Forest Test Performance")
print("=====================================")

print(f"MAE : {tuned_mae:.4f}")
print(f"RMSE: {tuned_rmse:.4f}")
print(f"R²  : {tuned_r2:.4f}")


# =========================
# Average Results
# =========================

average_mae = sum(rf_mae_scores) / len(rf_mae_scores)
average_rmse = (sum(rf_rmse_scores) /len(rf_rmse_scores))
average_r2 = sum(rf_r2_scores) / len(rf_r2_scores)

print("\nAverage Validation Performance")
print("===============================")

print(f"Average MAE : {average_mae:.4f}")
print(f"Average RMSE: {average_rmse:.4f}")
print(f"Average R²  : {average_r2:.4f}")

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
        "Random Forest",
        "Tuned Random Forest"
    ],
    "MAE": [
        baseline_mae,
        linear_mae,
        rf_mae,
        tuned_mae
    ],
    "RMSE": [
        baseline_rmse,
        linear_rmse,
        rf_rmse,
        tuned_rmse
    ],
    "R2": [
        baseline_r2,
        linear_r2,
        rf_r2,
        tuned_r2
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
# Top Feature
# =========================
top_features = importance.head(10).index.tolist()
print("\nTop 10 Features")
print("================")

for feature in top_features:
    print(feature)
X_top = stock[top_features]
X_top_train = X_top.iloc[:split_index]
X_top_test = X_top.iloc[split_index:]

top_model = RandomForestRegressor(
    **grid_search.best_params_,
    random_state=42,
    n_jobs=-1
)

top_model.fit(X_top_train,y_train)
top_predictions = top_model.predict(X_top_test)
top_mae = mean_absolute_error(y_test,top_predictions)
top_rmse = mean_squared_error(y_test,top_predictions) ** 0.5
top_r2 = r2_score(y_test,top_predictions)
print("\nTop Feature Model")
print("=================")
print(f"MAE : {top_mae:.4f}")
print(f"RMSE: {top_rmse:.4f}")
print(f"R²  : {top_r2:.4f}")

feature_comparison = pd.DataFrame({
    "Model": [
        "All Features",
        "Top Features"
    ],
    "MAE": [
        tuned_mae,
        top_mae
    ],
    "RMSE": [
        tuned_rmse,
        top_rmse
    ],
    "R2": [
        tuned_r2,
        top_r2
    ]
})

print("\nFeature Selection Comparison")
print("============================")

print(feature_comparison.to_string(index=False))
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

plt.plot(
    y_test.index,
    tuned_predictions,
    label="Tuned Random Forest"
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

importance.sort_values().plot(
    kind="barh",
    figsize=(12, 8)
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.grid(True)
plt.show()