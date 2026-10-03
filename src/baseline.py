import pandas as pd

def persistence(
        data: pd.DataFrame
    ) -> pd.Series:
    return data["PrecipIndicator"]

def climatology(
        data: pd.DataFrame, train_end: str = "2020"
        ) -> pd.Series:
    train = data.loc[:train_end]
    rain_rate_per_month = train["PrecipIndicator"].groupby(train.index.month).mean()
    return pd.Series(data.index.month.map(rain_rate_per_month), index = data.index)