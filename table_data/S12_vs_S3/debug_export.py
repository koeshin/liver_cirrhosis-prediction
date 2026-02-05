import os
print("Starting debug_export.py")
try:
    os.makedirs("artifacts/final_models", exist_ok=True)
    with open("artifacts/final_models/debug.txt", "w") as f:
        f.write("It worked!")
    print("Created directory and file.")
except Exception as e:
    print(f"Error: {e}")
