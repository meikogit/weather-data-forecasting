"""
Zentrale einstellungen des Projekts

Jede Entscheidung wie Stations-ID oder Start Datum steht hier 
"""
from pathlib import Path

# --- Folders -----------------------------------------------------------
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
