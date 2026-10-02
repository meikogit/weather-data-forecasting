import numpy as np
import pandas as pd 

def create_features(
        data: pd.DataFrame
):
    features_dict = {}


    features_dict["DewPointSpread"] = data["Temperature"] - data["DewPoint"]
    features_dict["d_Temperature_3h"] = data["Temperature"].diff(3)
    features_dict["Temperature"] = data["Temperature"]
    features_dict["d_NormalisedPressure_3h"] = data["NormalisedPressure"].diff(3)
    features_dict["NormalisedPressure"] = data["NormalisedPressure"]
    features_dict["RelativeHumidity"] = data["RelativeHumidity"]
    features_dict["d_RelativeHumidity_3h"] = data["RelativeHumidity"].diff(3)
    features_dict["DegOfCloudiness"] = data["DegOfCloudiness"]
    features_dict["MeanWindVeloc"] = data["MeanWindVeloc"]
    features_dict["PrecipIndicator"] = data["PrecipIndicator"]
    features_dict["d_DewPointSpread_3h"] = features_dict["DewPointSpread"].diff(3)
    features_dict["RollingDewPointSpread"] = features_dict["DewPointSpread"].rolling(3).mean()



    df_features = pd.DataFrame(features_dict)
    df_features["Month"] = data.index.month
    df_features["Month"] = df_features["Month"].replace([3, 4, 5], "F")
    df_features["Month"] = df_features["Month"].replace([6, 7, 8], "S")
    df_features["Month"] = df_features["Month"].replace([9, 10, 11], "H")
    df_features["Month"] = df_features["Month"].replace([12, 1, 2], "W")
    month_dummies = pd.get_dummies(
        df_features["Month"],
        prefix = "Season",
        dtype = int
    )
    df_features = pd.concat([df_features, month_dummies], axis = 1)
    df_features.drop(columns = "Month", inplace = True)
    return df_features