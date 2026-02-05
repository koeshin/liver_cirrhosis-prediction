import os

dirs = [
    "data/processed",
    "artifacts/best_params",
    "artifacts/oof_predictions",
    "artifacts/scores",
    "artifacts/optuna_studies",
    "artifacts/logs"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created {d}")
