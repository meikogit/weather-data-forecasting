import pandas as pd

def persistence(
        data: pd.DataFrame
    ) -> pd.Series:
    """
    Persistence forecast: it will rain in the future
    exactly when it is raining now.
    Returns 1 for rain and 0 for no rain.
    """
    return data["PrecipIndicator"]

def climatology(data: pd.DataFrame, train_end: str = "2020") -> pd.Series:
    """
    Climatology forecast: the rain probability is always the average
    share of rain hours in that month, learned from the training period only.
    Returns a probability between 0 and 1 for every hour.
    """
    train = data.loc[:train_end]
    rain_rate_per_month = train["PrecipIndicator"].groupby(train.index.month).mean()
    return pd.Series(data.index.month.map(rain_rate_per_month), index = data.index)