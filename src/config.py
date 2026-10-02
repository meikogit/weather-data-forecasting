"""
Central settings of the project.

Everything that is a decision (which station, which forecast horizons,
where the data is split in time) lives here.
"""
from pathlib import Path

# --- Folders -----------------------------------------------------------
# PROJECT_ROOT is the folder that contains "src/". Using it means the code
# finds its files no matter from which folder you start it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# --- Data --------------------------------------------------------------
STATION_ID = 164              # DWD station (see the file names in data/raw)
DATA_START = "2000-01-01"     # earlier years are not used

# --- Forecast horizons (in hours) ---------------------------------------
HORIZONS = [1, 2, 3, 4, 5, 6, 8, 12, 14, 16, 18, 24]

# --- Constants ----------------------------------------------------
SEED = 42
THRESHOLD = 0.3
