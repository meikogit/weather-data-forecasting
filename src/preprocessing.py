import pandas as pd 
import numpy as np

def process_df(
    df: pd.DataFrame
    )-> pd.DataFrame:
    """
    Process the DataFrame.
    """
    df = df.copy()
    try:
        df.index = df["MESS_DATUM"]
    except KeyError as error:
        raise ValueError(
            f"The Column does not exist. The Datetime Columsn is not: MESS_DATUM "
            ) from error

    try:
        df.index = pd.to_datetime(
        df.index.astype(str), 
        format = "%Y%m%d%H",
        )
    except ValueError as error:
        raise ValueError(
            "Date format does not match expected format YYYYMMDDHH."
        ) from error

    if "eor" in df.columns:
        df.drop("eor", inplace = True, axis = 1)

    try:
        df.rename(
        columns = {
            "STATIONS_ID": "StationID",
        },
        inplace = True
        )
    except ValueError as error:
        raise ValueError(
            f"Column not found: STATIONS_ID. Search for the Columnname and rename it"
            ) from error
            
    df.index.name = "Date"
    df.replace(-999, np.nan, inplace = True)
    df.columns = df.columns.str.strip()
    return df  
           

def clean_tu(
        df: pd.DataFrame
    )-> pd.DataFrame:
    """
    Process the DataFrame of Temperature and Humidity.
    """
    df = df.copy()

    try:
        df.rename(
        columns = {
            "QN_9": "QualityLevel_TH",
            "TT_TU": "Temperature",
            "RF_TU": "RelativeHumidity"
        },
        inplace = True
        )
    except ValueError as error:
        raise ValueError(
            f"One of the Columns not found: [QN_9, TT_TU, RF_TU]"
            ) from error

    return df  


def clean_p0(
        df: pd.DataFrame
    )-> pd.DataFrame:
    """
    Process the DataFrame of Pressure.
    """
    df = df.copy()

    try:
        df.rename(
        columns = {
            "QN_8": "QualityLevel_P",
            "P": "Pressure",
            "P0": "NormalisedPressure"
        },
        inplace = True
        )
    except ValueError as error:
        raise ValueError(
            f"One of the Columns not found: [QN_8, P, P0]"
            ) from error
    if "Pressure" in df.columns:
        df.drop("Pressure", inplace = True, axis = 1)
    return df  

def clean_rr(
        df: pd.DataFrame
    )-> pd.DataFrame:
    """
    Process the DataFrame of Precipitation Data.
    """
    df = df.copy()

    try:
        df.rename(
        columns = {
            "QN_8": "QualityLevel_R",
            "R1": "PrecipHight",
            "RS_IND": "PrecipIndicator",
            "WRTR": "PrecipType"
        },
        inplace = True
        )
    except ValueError as error:
        raise ValueError(
            f"One of the Columns not found: [QN_8, R1, RS_IND, WRTR]"
            ) from error
    return df  

def clean_ww(
        df: pd.DataFrame
    )-> pd.DataFrame:
    """
    Process the DataFrame of PhenomenonType Data.
    """
    df = df.copy()

    try:
        df.rename(
        columns = {
            "QN_8": "QualityLevel_WW",
            "WW": "PhenomenonType"
        },
        inplace = True
        )
    except ValueError as error:
        raise ValueError(
            f"One of the Columns not found: [QN_8, WW]"
            ) from error
    if "WW_Text" in df.columns:
            df.drop("WW_Text", inplace = True, axis = 1)
    return df  

def clean_td(
        df: pd.DataFrame
    )-> pd.DataFrame:
    """
    Process the DataFrame of DewPoint Data.
    """
    df = df.copy()

    try:
        df.rename(
        columns = {
            "QN_8": "QualityLevel_TD",
            "TD": "DewPoint"
        },
        inplace = True
        )
    except ValueError as error:
        raise ValueError(
            f"One of the Columns not found: [QN_8, TD]"
            ) from error
    if "TT" in df.columns:
        df.drop("TT", inplace = True, axis = 1)
    return df  

def clean_ff(
        df: pd.DataFrame
    )-> pd.DataFrame:
    """
    Process the DataFrame of Wind Data.
    """
    df = df.copy()

    try:
        df.rename(
        columns = {
           "QN_3": "QualityLevel_Wind",
            "F": "MeanWindVeloc",
            "D": "MeanWindDirect"
        },
        inplace = True
        )
    except ValueError as error:
        raise ValueError(
            f"One of the Columns not found: [QN_3, F, D]"
            ) from error
    return df  

def clean_n(
        df: pd.DataFrame
    )-> pd.DataFrame:
    """
    Process the DataFrame of Cloud Data.
    """
    df = df.copy()

    try:
        df.rename(
        columns = {
           "QN_8": "QualityLevel_Cloud",
            "V_N": "DegOfCloudiness"
        },
        inplace = True
        )
    except ValueError as error:
        raise ValueError(
            f"One of the Columns not found: [QN_8, V_N]"
            ) from error
    if "V_N_I" in df.columns:
        df.drop("V_N_I", axis = 1, inplace = True)
    return df  

def concat_columns(
        dfs: dict 
) -> pd.DataFrame:
    station_ids = [
       df["StationID"].iloc[0]
       for df in dfs.values() 
    ]
    if len(set(station_ids)) == 1:
        df_all = pd.concat(
            dfs.values(),
            axis=1,
            join="outer"
        )
    else:
        raise ValueError("StationID is not the same in all DataFrames.")
    return df_all


def clean_structure(
        df: pd.DataFrame
)-> pd.DataFrame:
    df = df.copy()
    df = df.sort_index()
    df = df.asfreq("h")
    return df