import json
import numpy as np
import os
import pandas as pd
from data import X, y, headlines
from model import training

# Create results folder
os.makedirs("../results", exist_ok=True)

# Train model and get results
metrics, confusion = training(X, y, headlines)

# Save metrics
metrics_path = "../results/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

# Save confusion matrix
confusion_path = "../results/confusion_matrix.csv"
pd.DataFrame(confusion).to_csv(confusion_path, index=False)

print("\nSaved results to:")
print(f" - {metrics_path}")
print(f" - {confusion_path}")
