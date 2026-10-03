import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src.config import SEED

def gradient_boosting(
        data: pd.DataFrame,
        features: list,
        indicator: pd.Series
    ):
    """
    Gibt das Gradient Boosting Modell gefittet auf die Trainingsdaten zurück
    """
    data_model = pd.concat(
            [data[features], indicator.rename("target")], axis = 1
            )
    data_model.dropna(inplace = True)

    X = data_model[features]
    y = data_model["target"]

    X_train = X.loc[:"2020", features]
    y_train = y.loc[:"2020"]

    model = HistGradientBoostingClassifier(random_state = SEED).fit(X_train, y_train)
    return model
    