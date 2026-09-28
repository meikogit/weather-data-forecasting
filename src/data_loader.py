import pandas as pd
from pathlib import Path

def load_data(
    filepath: str | Path,
    sep: str = ";",
    encoding: str = "ISO-8859-1",
              ) -> pd.DataFrame:
    """ 
    Loads a CSV or TXT file and returns it as a pandas DataFrame
    """
    filepath = Path(filepath)
    #1. Test for existance
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    #2. Test für right type of file
    if filepath.suffix.lower() not in [".txt", ".csv"]:
        raise ValueError( 
            f"Unsupported file Type: {filepath.suffix}."
            "Expected .csv or .txt."
                         )

    #3. Datei einlesen
    try:
        df = pd.read_csv(
        filepath, 
        sep = sep, 
        encoding = encoding
            )
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"File is empty: {filepath}") from error
    except pd.errors.ParserError as error:
        raise ValueError(f"Could not parse file: {filepath}") from error
    except UnicodeDecodeError as error:
        raise ValueError(f"Could not decode file: {filepath}. Check encoding") from error
    
    return df

   
   
