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

# --- Time split ---------------------------------------------------------
# Model is fitted on "train", decisions (features, threshold, settings) are
# made on "validation", and "test" is looked at only once at the very end.
# The split is always by time, never random.
TRAIN_END = "2017-12-31"      # train:      DATA_START ... TRAIN_END
VAL_END = "2023-12-31"        # validation: TRAIN_END  ... VAL_END
TEST_END = None               # test:       VAL_END    ... TEST_END

# --- Reproducibility ----------------------------------------------------
SEED = 42
THRESHOLD = 0.3
