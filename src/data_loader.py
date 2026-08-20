import yfinance as yf
import pandas as pd


def load_stock_data(file_path):

    stock = pd.read_csv(
        file_path,
        skiprows=[1, 2]
    )

    stock["Date"] = pd.to_datetime(
        stock["Price"]
    )

    stock["Close"] = pd.to_numeric(
        stock["Close"],
        errors="coerce"
    )

    stock["High"] = pd.to_numeric(
        stock["High"],
        errors="coerce"
    )

    stock["Low"] = pd.to_numeric(
        stock["Low"],
        errors="coerce"
    )

    stock["Open"] = pd.to_numeric(
        stock["Open"],
        errors="coerce"
    )

    stock["Volume"] = pd.to_numeric(
        stock["Volume"],
        errors="coerce"
    )

    stock = stock.dropna()

    stock = stock.set_index("Date")

    stock = stock.sort_index()

    return stock


def download_stock_data(
    symbol,
    period="2y"
):

    stock = yf.download(
        symbol,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if stock.empty:
        raise ValueError(
            f"No data found for {symbol}"
        )

    # Handle possible MultiIndex columns
    if isinstance(
        stock.columns,
        pd.MultiIndex
    ):
        stock.columns = (
            stock.columns.get_level_values(0)
        )

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    stock = stock[
        required_columns
    ]

    stock = stock.dropna()

    stock = stock.sort_index()

    return stock