import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV


def train_regression_model(X_train, y_train):

    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [
            100,
            200,
            300
        ],

        "max_depth": [
            None,
            10,
            20
        ],

        "min_samples_leaf": [
            1,
            2,
            4
        ]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    print(
        "Best parameters:",
        grid_search.best_params_
    )

    return grid_search.best_estimator_


def train_classification_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def save_model(model, file_path):

    joblib.dump(
        model,
        file_path
    )

    print(
        f"Model saved to: {file_path}"
    )

if __name__ == "__main__":

    import os

    from src.data_loader import load_stock_data
    from src.features import (
        create_features,
        create_target,
        FEATURES
    )

    print("Loading historical data...")

    stock = load_stock_data(
        "data/AAPL.csv"
    )

    print("Creating features...")

    stock = create_features(stock)

    stock = create_target(stock)

    stock = stock.dropna()

    X = stock[FEATURES]

    y_price = stock["Target"]

    y_direction = stock["Direction"]

    split_index = int(
        len(stock) * 0.8
    )

    X_train = X.iloc[:split_index]

    y_price_train = y_price.iloc[
        :split_index
    ]

    y_direction_train = y_direction.iloc[
        :split_index
    ]

    print(
        "\nTraining price model..."
    )

    price_model = train_regression_model(
        X_train,
        y_price_train
    )

    print(
        "\nTraining direction model..."
    )

    direction_model = train_classification_model(
        X_train,
        y_direction_train
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    save_model(
        price_model,
        "models/price_model.pkl"
    )

    save_model(
        direction_model,
        "models/direction_model.pkl"
    )

    print(
        "\nTraining completed."
    )