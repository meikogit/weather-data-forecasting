import pandas as pd

def persistence(
        data: pd.DataFrame
    ) -> pd.Series:
    """
    Das Persistenz Model geht davon aus, dass das Wetter gleich bleibt wie es aktuell ist
    """
    return data["PrecipIndicator"]

def climatology(
        data: pd.DataFrame, train_end: str = "2020"
        ) -> pd.Series:
    """
    Das Klimatologie Modell schaut wie oft es in dem jeweiligen Monat die Jahre zuvor 
    geregnet hat und nimmt den Anteil als Regenwahrscheinlichkeit
    """
    train = data.loc[:train_end]
    rain_rate_per_month = train["PrecipIndicator"].groupby(train.index.month).mean()
    return pd.Series(data.index.month.map(rain_rate_per_month), index = data.index)