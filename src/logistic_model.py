import pandas as pd
import statsmodels.api as sm


def logistic_reg(
        data: pd.DataFrame,
        features: list,
        indicator: pd.Series
):
    """
    Gibt das Logistosche Regressions Modell zurück, gefittet auf Trainingsdaten
    """
    data_model = pd.concat(
        [data[features], indicator.rename("target")], axis = 1
        )
    data_model.dropna(inplace = True)
    X = data_model[features]
    y = data_model["target"]

    X_train = X.loc[:"2020", features]
    y_train = y.loc[:"2020"]

    X_train = sm.add_constant(X_train)
    
    
    model = sm.Logit(y_train, X_train).fit()

    return model

