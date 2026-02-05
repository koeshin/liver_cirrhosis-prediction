import os

dirs = [
    "data/processed",
    "artifacts/best_params",
    "artifacts/oof_predictions",
    "artifacts/scores",
    "artifacts/optuna_studies"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created {d}")

with open("setup_verified.txt", "w") as f:
    f.write("Setup ran successfully!")
