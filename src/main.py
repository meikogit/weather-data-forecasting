from src.config import RAW_DIR, RESULTS_DIR, HORIZONS, THRESHOLD
from src.data_loader import load_data
from src.features import create_features
from src.evaluation import evaluate
from src.logistic_model import logistic_reg
from src.gradient_boosting import gradient_boosting
import pandas as pd
from src import preprocessing
from src.baseline import persistence, climatology
from src.evaluation import evaluate_predictions, onset_recall
from sklearn.metrics import brier_score_loss
from src.plots import plot_brier, plot_onset_recall

def main():

    #Load and Preprocess Data
    
    dfs = {}

    for path in RAW_DIR.glob("produkt_*_stunde_*.txt"):
        name = path.stem.split("produkt_")[1].split("_stunde")[0]
        dfs[name] = preprocessing.process_df(load_data(path))

    #Clean the column names of each variable
    dfs["tu"] = preprocessing.clean_tu(dfs["tu"])
    dfs["p0"] = preprocessing.clean_p0(dfs["p0"])
    dfs["ff"] = preprocessing.clean_ff(dfs["ff"])
    dfs["rr"] = preprocessing.clean_rr(dfs["rr"])
    dfs["td"] = preprocessing.clean_td(dfs["td"])
    dfs["ww"] = preprocessing.clean_ww(dfs["ww"])
    dfs["n"] = preprocessing.clean_n(dfs["n"])

    #Concatenate DataFrames
    data = preprocessing.concat_columns(dfs)

    #Explenation of the columns
    explained_columns = {
            "Temperature": "Temperature in degrees celsius",
            "RelativeHumidity": "The amount of water vapor in the air relative to the maximum amount the air can hold at the same temperature",
            "NormalisedPressure": "Pressure normalised on sealevel",
            "PrecipHight": "Hight of precipitation in mm",
            "PrecipIndicator": "1 for rain and 0 for no rain",
            "PrecipType": "0: no rain, 6: liquid, 7: solid (snow), 8: mixed",
            "PhenomenonType": "More specific explenation of phenomenon",
            "DewPoint": "Temperature in degrees celsius where the air with same pressure is saturated with vapor",
            "MeanWindVeloc": "Mean velocity of the wind",
            "MeanWindDirect": "Mean direction of the wind in degrees",
            "DegOfCloudiness": "Amount of clouds in steps from 0 to 8, where 0 stands for cloud free sky."
        }

    data = preprocessing.clean_structure(data)
    data = data.loc["2000":]

    #Features
    features = ['DewPointSpread', 'd_Temperature_3h', 'Temperature',
       'd_NormalisedPressure_3h', 'NormalisedPressure',
       'd_RelativeHumidity_3h', 'DegOfCloudiness',
       'MeanWindVeloc', 'PrecipIndicator', 'Season_H',
       'Season_W', "Season_F", "d_DewPointSpread_3h", "RollingDewPointSpread"]
    df_features = create_features(data)
    forecast_hours = HORIZONS
    results = []
    p_pred_log = {}
    p_pred_grad = {}

    #Climatology baseline: same probabilities for every horizon, so compute once
    p_clima = climatology(data)
    
    
    for hours in forecast_hours:
        n = hours
        indicator_nh = data["PrecipIndicator"].shift(-n)

        #Logistic Regression Model
        logit_model = logistic_reg(data = df_features, indicator = indicator_nh, features = features )
    
        #Evaluation Logistic Regression
        evaluation_logit = evaluate(data = df_features[features], indicator = indicator_nh, threshold = THRESHOLD, model = logit_model)

        #Gradient Boosting
        grad_model = gradient_boosting(data = df_features, indicator = indicator_nh, features = features)

        #Evaluation Gradient Boosting
        evaluation_grad = evaluate(data = df_features[features], indicator = indicator_nh, threshold = THRESHOLD, model = grad_model)

        #Persistence baseline 
        test_index = evaluation_logit["p_pred"].index
        persistence_scores = evaluate_predictions(
            y_true = indicator_nh.loc[test_index],
            y_pred = persistence(data).loc[test_index]
        )

        #Climatology baseline (only Brier, because it gives probabilities)
        climatology_brier = brier_score_loss(
            indicator_nh.loc[test_index],
            p_clima.loc[test_index]
        )
    
        #Rain onsets: how many rain starts does the model predict?
        y_pred_log = (evaluation_logit["p_pred"] >= THRESHOLD).astype(int)
        model_onset_recall_log = onset_recall(
            now = data["PrecipIndicator"].loc[test_index],
            y_true = indicator_nh.loc[test_index],
            y_pred = y_pred_log,
        )

        y_pred_grad = (evaluation_grad["p_pred"] >= THRESHOLD).astype(int)
        model_onset_recall_grad = onset_recall(
            now = data["PrecipIndicator"].loc[test_index],
            y_true = indicator_nh.loc[test_index],
            y_pred = y_pred_grad,
        )


        results.append({
        "Hours": hours,
        "ROC-AUC_log": evaluation_logit["ROC-AUC"],
        "ROC-AUC_grad": evaluation_grad["ROC-AUC"],
        "F1 Score_log": evaluation_logit["F1_Score"],
        "F1 Score_grad": evaluation_grad["F1_Score"],
        "Brier_log": evaluation_logit["Brier"],
        "Brier_grad": evaluation_grad["Brier"],
        "Persistence Accuracy": persistence_scores["Accuracy"],
        "Persistence F1": persistence_scores["F1 Score"],
        "Persistence Brier": persistence_scores["Brier"],
        "Climatology Brier": climatology_brier,
        "Onset Recall_log": model_onset_recall_log,
        "Onset Recall_grad": model_onset_recall_grad
        })
        p_pred_log[hours] = evaluation_logit["p_pred"]
        p_pred_grad[hours] = evaluation_grad["p_pred"]
        
    results_df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents = True, exist_ok = True)
    plot_brier(results_df, RESULTS_DIR / "brier_score.png")
    plot_onset_recall(results_df, RESULTS_DIR / "onset_recall.png")
    with open(RESULTS_DIR / "results_logistic_reg.txt", "w") as file:
        file.write("LOGISTIC REGRESSION - FORECAST HORIZON EVALUATION\n")
        file.write("=" * 70 + "\n\n")
        file.write("Model: Logistic Regression\n")
        file.write("Training period: 2000-2020\n")
        file.write("Test period: 2021-present\n")
        file.write(f"Threshold: {THRESHOLD} \n\n")
        file.write("Features:\n")
        for feature in features:
            file.write(f"  - {feature}\n")
        file.write(
            results_df.to_string(
            index = False, 
            float_format=lambda x: f"{x:.3f}" 
            )
            )
        
    print("\nForecast Horizon Evaluation")
    print(results_df.to_string(index=False))
    


    
if __name__ == "__main__":
    main()
    


