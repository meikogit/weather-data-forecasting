from src.config import RAW_DIR, RESULTS_DIR, HORIZONS
from src.data_loader import load_data, group_files_by_variable
from src.features import create_features
from src.evaluation import evaluate
from src.logistic_model import logistic_reg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src import preprocessing

def main():

    #Load and Preprocess Data
    #One variable (e.g. "tu") can consist of several files (historical + recent).
    #Each file is processed on its own, then the files of one variable are merged.
    dfs = {}
    for name, paths in group_files_by_variable(RAW_DIR).items():
        parts = [preprocessing.process_df(load_data(path)) for path in paths]
        dfs[name] = preprocessing.merge_files(parts)

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
       'd_RelativeHumidity_3h', 'RelativeHumidity', 'DegOfCloudiness',
       'MeanWindVeloc', 'PrecipIndicator', 'Season_H',
       'Season_W', "d_DewPointSpread_3h", "RollingDewPointSpread"]
    df_features = create_features(data)
    forecast_hours = HORIZONS
    results = []
    p_pred = {}
    thresholds = {
        "1": 0.4,
        "2": 0.4,
        "3": 0.4,
        "4": 0.4,
        "5": 0.3,
        "6": 0.3,
        "8": 0.2,
        "12": 0.2,
        "14": 0.2,
        "16": 0.2,
        "18": 0.2,
        "24": 0.2
    }
    for hours in forecast_hours:
        threshold = thresholds[str(hours)]
        n = hours
        indicator_nh = data["PrecipIndicator"].shift(-n)

        #Logistic Regression
        logit_model = logistic_reg(data = df_features, indicator = indicator_nh, features = features )
    
        #Evaluation
        evaluation = evaluate(data = df_features[features], indicator = indicator_nh, threshold = threshold, model = logit_model)

        results.append({
        "Hours": hours,
        "ROC-AUC": evaluation["ROC-AUC"],
        "Missed Rain": evaluation["MissedRain"],
        "False Rain": evaluation["FalseRain"],
        "Accuracy": evaluation["Accuracy"],
        "F1 Score": evaluation["F1_Score"]
        })
        p_pred[hours] = evaluation["p_pred"]
        
    results_df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents = True, exist_ok = True)
    with open(RESULTS_DIR / "results_logistic_reg.txt", "w") as file:
        file.write("LOGISTIC REGRESSION - FORECAST HORIZON EVALUATION\n")
        file.write("=" * 70 + "\n\n")
        file.write("Model: Logistic Regression\n")
        file.write("Training period: 2000-2020\n")
        file.write("Test period: 2021-present\n")
        file.write(f"Thresholds per horizon (hours: threshold): {thresholds}\n\n")
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
    


