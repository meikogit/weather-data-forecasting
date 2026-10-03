import statsmodels.api as sm 
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, brier_score_loss
from statsmodels.discrete.discrete_model import BinaryResultsWrapper



def evaluate(
        data: pd.DataFrame,
        indicator: pd.Series,
        threshold: int,
        model,
    )->dict:
    """
    Evaluiert das Modell indem es die vorhergsagten 
    Regenwahrscheinlichkeiten bestimmt und einige Vergleichswerte berechnet
    """
    evaluation = {}
    data_model = pd.concat(
    [data, indicator.rename("target")],
    axis=1
    ).dropna()
    test_data = data_model.loc["2021":]

    X_test = test_data.drop(columns="target")
    y_test = test_data["target"]

    if type(model) == BinaryResultsWrapper:
        X_test = sm.add_constant(X_test)
        p_pred = model.predict(X_test)
        evaluation["Summary"] = model.summary()
    else:
        p_pred = pd.Series(model.predict_proba(X_test)[:, 1], index = X_test.index)

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

    evaluation["Accuracy"] = accuracy
    evaluation["RainRecall"] = rain_recall
    evaluation["RainPrecision"] = rain_precision
    evaluation["MissedRain"] = missed_rain
    evaluation["FalseRain"] = false_rain
    evaluation["F1_Score"] = f1
    evaluation["ROC-AUC"] = roc_auc_score(y_test, p_pred)
    evaluation["p_pred"] = p_pred
    evaluation["ConfusionMatrix"] = matrix
    evaluation["Brier"] = brier_score_loss(y_test, p_pred)
    return evaluation 


def evaluate_predictions(
        y_true: pd.Series,
        y_pred: pd.Series,
) -> dict:
    """
    Evaluierung der Persistenz
    """

    f1 = f1_score(y_true, y_pred)
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1,
        "Brier": brier_score_loss(y_true, y_pred)
    }

def onset_recall(
        now: pd.Series,
        y_true: pd.Series,
        y_pred: pd.Series,
) -> float:
    """
    Anteil der erkannten Regenbeginne (Persistenz ist hier per Definition 0)
    """
    onset = (now == 0) & (y_true == 1)
    return y_pred[onset].mean()