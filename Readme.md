AAPL Stock Price Prediction

A machine learning project for predicting the next-day closing price of Apple Inc. (AAPL) using historical stock market data and engineered technical features.

The project compares a simple baseline prediction with Linear Regression and Random Forest Regression models.

Project Overview

The objective of this project is to investigate whether historical AAPL price and trading information can be used to predict the following day's closing price.

The project follows these main steps:

Load historical AAPL stock data.
Create technical and lag-based features.
Create a next-day closing price target.
Split the data chronologically into training and testing sets.
Establish a simple baseline prediction.
Train Linear Regression and Random Forest models.
Evaluate model performance using MAE, RMSE, and R².
Compare the models.
Analyze Random Forest feature importance.
Visualize the actual and predicted prices.
Features

The following features are created from the historical stock data.

Moving Averages
MA5 - 5-day moving average
MA20 - 20-day moving average
MA50 - 50-day moving average
Return and Volatility
Return - daily percentage return
Volatility20 - 20-day rolling standard deviation of returns
Lag Features
Lag1
Lag2
Lag3
Lag5
Lag10

These represent previous closing prices and provide the models with historical price information.

Lagged Returns
ReturnLag1
ReturnLag2
ReturnLag3

These represent previous daily returns.

Target

The prediction target is the next trading day's closing price:

stock["Target"] = stock["Close"].shift(-1)


Therefore, the model attempts to predict:

Tomorrow's closing price using information available today and earlier.

Machine Learning Models
1. Baseline

The baseline assumes that the next day's closing price will be equal to today's closing price.

baseline_predictions = X_test["Close"]


This provides a simple benchmark for determining whether the machine learning models provide an improvement over a naive prediction.

2. Linear Regression

A Linear Regression model is used as a simple machine learning benchmark.

linear_model = LinearRegression()


It attempts to model a linear relationship between the engineered features and the next-day closing price.

3. Random Forest Regression

A Random Forest Regressor is used to capture potentially nonlinear relationships between the features and the target.

The current model uses:

300 trees
Maximum depth of 12
Minimum 2 samples per leaf
Random state of 42
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

Train/Test Split

The dataset is divided chronologically:

80% - training data
20% - testing data

The data is not randomly shuffled because stock prices are time-series data. Maintaining chronological order helps prevent future observations from being used to train the model.

split_index = int(len(stock) * 0.8)

Evaluation Metrics

Three regression metrics are used.

Mean Absolute Error (MAE)

Measures the average absolute difference between the predicted and actual prices.

Lower MAE indicates better performance.

Root Mean Squared Error (RMSE)

Measures the square root of the average squared prediction error.

Lower RMSE indicates better performance and gives greater weight to larger errors.

R² Score

Measures how much of the variation in the target variable is explained by the model.

A higher R² generally indicates better performance.

Model Comparison

The project produces a comparison table containing:

Model	MAE	RMSE	R²
Baseline	Calculated by program	Calculated by program	Calculated by program
Linear Regression	Calculated by program	Calculated by program	Calculated by program
Random Forest	Calculated by program	Calculated by program	Calculated by program

The actual values depend on the dataset contained in data/AAPL_clean.csv.

Feature Importance

The Random Forest model provides feature importance values to identify which features contribute most strongly to its predictions.

The program prints the feature importance values and generates a bar chart for visualization.

This helps investigate which historical price, return, volatility, and moving-average features are most useful to the Random Forest model.

Visualizations

The project generates two visualizations.

Actual vs Predicted Prices

This chart compares:

Actual AAPL closing prices
Linear Regression predictions
Random Forest predictions
Random Forest Feature Importance

This chart displays the importance assigned to each feature by the Random Forest model.

Project Structure
project/
│
├── data/
│   └── AAPL_clean.csv
│
├── your_script.py
│
├── requirements.txt
│
└── README.md


Replace your_script.py with the actual filename of your Python script.

Dataset

The program expects a CSV file at:

data/AAPL_clean.csv


The dataset should contain at least the following columns:

Date
Close
Volume


Date is used as the DataFrame index and is parsed as a datetime value.

Installation

Clone or download the project and navigate to the project directory.

Create a virtual environment if desired:

python -m venv venv


Activate the virtual environment.

Windows
venv\Scripts\activate

macOS/Linux
source venv/bin/activate


Install the required packages:

pip install -r requirements.txt

Running the Project

Make sure the dataset is located at:

data/AAPL_clean.csv


Then run the Python script:

python your_script.py


The program will print:

Model comparison results
Random Forest feature importance

It will also display:

Actual vs predicted prices
Feature importance visualization
Limitations

This project has several limitations.

Stock prices are affected by many external factors that are not included in the dataset.
The model only uses historical market information.
The target is the next-day closing price, which can be strongly influenced by the current closing price.
An 80/20 chronological split provides only one final test period.
The current version does not perform hyperparameter tuning.
The current version does not use more advanced technical indicators such as RSI, MACD, or Bollinger Bands.
Prediction accuracy does not necessarily imply that a trading strategy would be profitable.
Future Improvements

Possible future improvements include:

Adding RSI, MACD, and Bollinger Bands.
Implementing TimeSeriesSplit cross-validation.
Adding GridSearchCV for hyperparameter tuning.
Testing additional machine learning models.
Predicting next-day returns instead of the raw closing price.
Adding directional classification for predicting whether the price will increase or decrease.
Performing walk-forward validation.
Evaluating a simple trading strategy based on model predictions.
Disclaimer

This project is intended for educational and research purposes only. The predictions generated by the models should not be considered financial advice or a recommendation to buy or sell securities.