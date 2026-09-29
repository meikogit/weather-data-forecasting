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


def group_files_by_variable(
    folder: str | Path,
) -> dict[str, list[Path]]:
    """
    Finds all DWD hourly files in a folder and groups them by variable.

    A file name looks like  produkt_tu_stunde_19560101_20251231_00164.txt
    Here "tu" is the variable (temperature/humidity). DWD delivers newer
    data in a second file for the same variable, so a variable can have
    several files. They are sorted from oldest to newest (the start date is
    part of the file name, so sorting by name sorts by time).
    """
    folder = Path(folder)
    groups: dict[str, list[Path]] = {}
    for path in sorted(folder.glob("produkt_*_stunde_*.txt")):
        name = path.stem.split("produkt_")[1].split("_stunde")[0]
        groups.setdefault(name, []).append(path)

    if not groups:
        raise FileNotFoundError(f"No DWD hourly files found in: {folder}")
    return groups
