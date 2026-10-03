import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
YELLOW = "#eda100"

def plot_brier(
    results_df: pd.DataFrame,
    path: Path    
)-> None:
    """
    Plottet den Brier Score der verschiedenen Modelle
    """

    fig, ax = plt.subplots(figsize = (8, 4.5))
    hours = results_df["Hours"]

    ax.plot(hours, results_df["Brier_log"], color = BLUE, marker = "o", linewidth = 2, label = "Logistische Regression")
    ax.plot(hours, results_df["Brier_grad"], color = ORANGE, marker = "s", linewidth = 2, label = "Gradient Boosting")
    ax.plot(hours, results_df["Persistence Brier"], color = GREEN, marker = "^", linewidth = 2, linestyle = "--", label = "Persistenz")
    ax.plot(hours, results_df["Climatology Brier"], color = YELLOW, linewidth = 2, linestyle = ":", label = "Klimatologie")

    ax.set_title("Brier Score nach Vorhersagehorizont (kleiner = besser)")
    ax.set_xlabel("Vorhersagehorizont in Stunden")
    ax.set_ylabel("Brier Score")
    ax.set_xticks(hours)
    ax.grid(alpha = 0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon = False)

    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    plt.close(fig)

def plot_onset_recall(
        results_df: pd.DataFrame, path: Path
        ) -> None:
    """
    Plottet den Anteil der erkannten Regenbeginne von Gradient Boosting und Logistic Regression
    """
    fig, ax = plt.subplots(figsize = (8, 4.5))
    hours = results_df["Hours"]

    ax.plot(hours, results_df["Onset Recall_log"] * 100, color = BLUE, marker = "o", linewidth = 2, label = "Logistische Regression")
    ax.plot(hours, results_df["Onset Recall_grad"] * 100, color = ORANGE, marker = "s", linewidth = 2, label = "Gradient Boosting")

    ax.set_title("Erkannte Regenbeginne (Onset Recall, Schwellenwert 0,3)")
    ax.set_xlabel("Vorhersagehorizont in Stunden")
    ax.set_ylabel("Erkannte Regenbeginne in %")
    ax.set_xticks(hours)
    ax.set_ylim(bottom = 0)
    ax.grid(alpha = 0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon = False)

    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    plt.close(fig)