import joblib

from src.data_loader import download_stock_data

from src.features import (
    create_features,
    FEATURES
)


def load_model(file_path):

    return joblib.load(
        file_path
    )


def predict_price(
    model,
    features
):

    prediction = model.predict(
        features
    )

    return prediction[0]


def predict_direction(
    model,
    features
):

    prediction = model.predict(
        features
    )

    return prediction[0]


if __name__ == "__main__":

    symbol = "AAPL"

    print(
        f"Downloading latest {symbol} data..."
    )

    stock = download_stock_data(
        symbol,
        period="2y"
    )

    print(
        "Creating features..."
    )

    stock = create_features(
        stock
    )

    stock = stock.dropna()

    X = stock[FEATURES]

    latest_features = X.iloc[
        [-1]
    ]

    print(
        "Loading trained models..."
    )

    price_model = load_model(
        "models/price_model.pkl"
    )

    direction_model = load_model(
        "models/direction_model.pkl"
    )

    predicted_price = predict_price(
        price_model,
        latest_features
    )

    predicted_direction = predict_direction(
        direction_model,
        latest_features
    )

    current_price = (
        stock["Close"].iloc[-1]
    )

    if predicted_direction == 1:
        direction_text = "UP"
    else:
        direction_text = "DOWN"

    print(
        "\n============================"
    )

    print(
        f"{symbol} STOCK PREDICTION"
    )

    print(
        "============================"
    )

    print(
        f"Current Price: "
        f"${current_price:.2f}"
    )

    print(
        f"Predicted Price: "
        f"${predicted_price:.2f}"
    )

    print(
        f"Predicted Direction: "
        f"{direction_text}"
    )

    print(
        "============================"
    )