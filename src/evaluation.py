import statsmodels.api as sm 
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def evaluate(
        data: pd.DataFrame,
        indicator: pd.Series,
        threshold: int,
        model,
)->dict:
    evaluation = {}
    data_model = pd.concat(
    [data, indicator.rename("target")],
    axis=1
    ).dropna()
    test_data = data_model.loc["2021":]

    X_test = test_data.drop(columns="target")
    y_test = test_data["target"]

    X_test = sm.add_constant(X_test)

    p_pred = model.predict(X_test)
    y_pred = (p_pred >= threshold).astype(int)
    
    matrix = confusion_matrix(y_test, y_pred)

    accuracy = (y_test == y_pred).mean()
    TN = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    TP = matrix[1, 1]

    missed_rain = FN / (TP + FN)
    false_rain = FP / (TN + FP)

    rain_recall = TP / (TP + FN)
    rain_precision = TP / (TP + FP)

    f1 = (
        2 * rain_precision * rain_recall /
        (rain_precision + rain_recall)
    )

    evaluation["Summary"] = model.summary()
    evaluation["Accuracy"] = accuracy
    evaluation["RainRecall"] = rain_recall
    evaluation["RainPrecision"] = rain_precision
    evaluation["MissedRain"] = missed_rain
    evaluation["FalseRain"] = false_rain
    evaluation["F1_Score"] = f1
    evaluation["ROC-AUC"] = roc_auc_score(y_test, p_pred)
    evaluation["p_pred"] = p_pred
    evaluation["ConfusionMatrix"] = matrix
    return evaluation