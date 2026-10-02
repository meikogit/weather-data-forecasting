import pandas as pd

def persistence(data: pd.DataFrame) -> pd.Series:
    """
    Persistence forecast: it will rain in the future
    exactly when it is raining now.
    Returns 1 for rain and 0 for no rain.
    """
    return data["PrecipIndicator"]
