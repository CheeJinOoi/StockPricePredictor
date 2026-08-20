def create_features(stock):

    stock = stock.copy()

    # Moving averages
    stock["MA5"] = (
        stock["Close"]
        .rolling(5)
        .mean()
    )

    stock["MA20"] = (
        stock["Close"]
        .rolling(20)
        .mean()
    )

    stock["MA50"] = (
        stock["Close"]
        .rolling(50)
        .mean()
    )

    # Daily return
    stock["Return"] = (
        stock["Close"]
        .pct_change()
    )

    # Volatility
    stock["Volatility20"] = (
        stock["Return"]
        .rolling(20)
        .std()
    )

    # Lag features
    stock["Lag1"] = (
        stock["Close"].shift(1)
    )

    stock["Lag2"] = (
        stock["Close"].shift(2)
    )

    stock["Lag3"] = (
        stock["Close"].shift(3)
    )

    stock["Lag5"] = (
        stock["Close"].shift(5)
    )

    stock["Lag10"] = (
        stock["Close"].shift(10)
    )

    # Lagged returns
    stock["ReturnLag1"] = (
        stock["Return"].shift(1)
    )

    stock["ReturnLag2"] = (
        stock["Return"].shift(2)
    )

    stock["ReturnLag3"] = (
        stock["Return"].shift(3)
    )

    # RSI
    delta = stock["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = (
        gain.rolling(14).mean()
    )

    average_loss = (
        loss.rolling(14).mean()
    )

    rs = (
        average_gain / average_loss
    )

    stock["RSI"] = (
        100 - (100 / (1 + rs))
    )

    # MACD
    ema12 = (
        stock["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        stock["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    stock["MACD"] = (
        ema12 - ema26
    )

    stock["MACD_Signal"] = (
        stock["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    stock["MACD_Hist"] = (
        stock["MACD"]
        - stock["MACD_Signal"]
    )

    # Bollinger Bands
    bb_middle = (
        stock["Close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        stock["Close"]
        .rolling(20)
        .std()
    )

    stock["BB_Upper"] = (
        bb_middle + 2 * bb_std
    )

    stock["BB_Lower"] = (
        bb_middle - 2 * bb_std
    )

    stock["BB_Position"] = (
        (stock["Close"] - stock["BB_Lower"])
        / (stock["BB_Upper"] - stock["BB_Lower"])
    )

    return stock
FEATURES = [
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
def create_target(stock):

    stock = stock.copy()

    stock["Target"] = (
        stock["Close"].shift(-1)
    )

    stock["Direction"] = (
        stock["Target"] > stock["Close"]
    ).astype(int)

    return stock