import json
import numpy as np
import os
import pandas as pd
from data import X, y
from model import training

# make results folder and train
os.makedirs("../results", exist_ok=True)
metrics, confusion = training(X, y)

# Save metrics
metrics_path = "../results/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

# Save confusion matrix
confusion_path = "../results/confusion_matrix.csv"
pd.DataFrame(confusion).to_csv(confusion_path, index=False)

print("\n Saved results to: ")
print(f" - {metrics_path}")
print(f" - {confusion_path}")
