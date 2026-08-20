import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def create_backtest_results(
    stock,
    predictions
):

    results = stock.copy()

    results["Prediction"] = predictions

    results["Market_Return"] = (
        results["Close"]
        .pct_change()
    )

    results["Strategy_Return"] = (
        results["Market_Return"]
        * results["Prediction"]
        .shift(1)
    )

    return results

def calculate_cumulative_returns(
    results
):

    results["Market_Equity"] = (
        1
        + results["Market_Return"]
        .fillna(0)
    ).cumprod()

    results["Strategy_Equity"] = (
        1
        + results["Strategy_Return"]
        .fillna(0)
    ).cumprod()

    return results

def calculate_total_return(
    equity_curve
):

    return (
        equity_curve.iloc[-1]
        - 1
    )

def calculate_max_drawdown(
    equity_curve
):

    peak = equity_curve.cummax()

    drawdown = (
        equity_curve - peak
    ) / peak

    return drawdown.min()

def calculate_annualized_return(
    equity_curve,
    trading_days=252
):

    total_days = len(
        equity_curve
    )

    if total_days <= 1:
        return 0

    total_return = (
        equity_curve.iloc[-1]
        / equity_curve.iloc[0]
    )

    years = (
        total_days
        / trading_days
    )

    return (
        total_return
        ** (1 / years)
        - 1
    )

def calculate_sharpe_ratio(
    strategy_returns,
    trading_days=252
):

    daily_returns = (
        strategy_returns
        .dropna()
    )

    if daily_returns.std() == 0:
        return 0

    return (
        daily_returns.mean()
        / daily_returns.std()
    ) * np.sqrt(trading_days)

def evaluate_strategy(
    results
):

    strategy_return = calculate_total_return(
        results["Strategy_Equity"]
    )

    market_return = calculate_total_return(
        results["Market_Equity"]
    )

    annualized_return = calculate_annualized_return(
        results["Strategy_Equity"]
    )

    max_drawdown = calculate_max_drawdown(
        results["Strategy_Equity"]
    )

    sharpe_ratio = calculate_sharpe_ratio(
        results["Strategy_Return"]
    )

    return {
        "strategy_return": strategy_return,
        "market_return": market_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio
    }

def generate_walk_forward_predictions(
    X,
    y_direction,
    train_size,
    model_factory
):

    predictions = []

    prediction_dates = []

    for i in range(
        train_size,
        len(X)
    ):

        X_train = X.iloc[:i]

        y_train = y_direction.iloc[:i]

        X_current = X.iloc[
            [i]
        ]

        model = model_factory()

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_current
        )[0]

        predictions.append(
            prediction
        )

        prediction_dates.append(
            X.index[i]
        )

    return pd.Series(
        predictions,
        index=prediction_dates
    )

def create_direction_model():

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
model_factory=create_direction_model

def add_transaction_costs(
    results,
    transaction_cost=0.001
):

    position_change = (
        results["Prediction"]
        .diff()
        .abs()
    )

    costs = (
        position_change
        * transaction_cost
    )

    results["Strategy_Return_After_Cost"] = (
        results["Strategy_Return"]
        - costs
    )

    return results

def calculate_cost_adjusted_equity(
    results
):

    results["Strategy_Equity_After_Cost"] = (
        1
        + results[
            "Strategy_Return_After_Cost"
        ].fillna(0)
    ).cumprod()

    return results

def calculate_classification_metrics(
    actual,
    predicted
):

    accuracy = accuracy_score(
        actual,
        predicted
    )

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    matrix = confusion_matrix(
        actual,
        predicted
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix
    }

def calculate_win_rate(
    strategy_returns
):

    valid_returns = (
        strategy_returns
        .dropna()
    )

    if len(valid_returns) == 0:
        return 0

    winning_trades = (
        valid_returns > 0
    ).sum()

    return (
        winning_trades
        / len(valid_returns)
    )

def calculate_drawdown_curve(
    equity_curve
):

    peak = equity_curve.cummax()

    drawdown = (
        equity_curve - peak
    ) / peak

    return drawdown