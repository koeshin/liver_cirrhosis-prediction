import kagglehub

# Download latest version
path = kagglehub.dataset_download("aadarshvelu/liver-cirrhosis-stage-classification")

print("Path to dataset files:", path)