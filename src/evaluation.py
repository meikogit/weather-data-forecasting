import statsmodels.api as sm 
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score


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

    f1 = f1_score(y_test, y_pred)

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


def evaluate_predictions(
        y_true: pd.Series,
        y_pred: pd.Series,
) -> dict:
    """
    Scores a 0/1 forecast against what really happened.
    Works for every forecast.
    """

    f1 = f1_score(y_true, y_pred)
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1,
    }