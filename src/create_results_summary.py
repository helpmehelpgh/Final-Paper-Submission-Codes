import pandas as pd
from pathlib import Path

results = [
    {
        "Task": "Earthquake vs Noise Classification",
        "Training Type": "Balanced binary dataset",
        "Accuracy": 0.98,
        "Macro F1": 0.98,
        "Key Observation": "The model clearly distinguished earthquake and noise waveforms with very high accuracy."
    },
    {
        "Task": "Magnitude Class Prediction",
        "Training Type": "Imbalanced original dataset",
        "Accuracy": 0.88,
        "Macro F1": 0.67,
        "Key Observation": "High overall accuracy, but biased toward the dominant M<2 class."
    },
    {
        "Task": "Magnitude Class Prediction",
        "Training Type": "Balanced dataset",
        "Accuracy": 0.70,
        "Macro F1": 0.70,
        "Key Observation": "Lower accuracy, but more reliable performance across all magnitude classes."
    },
    {
        "Task": "Distance Class Prediction",
        "Training Type": "Imbalanced original dataset",
        "Accuracy": 0.85,
        "Macro F1": 0.56,
        "Key Observation": "High overall accuracy, but weak performance for far-distance records."
    },
    {
        "Task": "Distance Class Prediction",
        "Training Type": "Balanced dataset",
        "Accuracy": 0.72,
        "Macro F1": 0.72,
        "Key Observation": "Balanced training improved recognition of far-distance earthquake records."
    }
]

out_dir = Path("results/tables")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(results)
df.to_csv(out_dir / "results_summary.csv", index=False)

print(df)
print("\nSaved:")
print(out_dir / "results_summary.csv")
