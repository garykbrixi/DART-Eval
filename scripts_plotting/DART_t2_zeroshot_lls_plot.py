import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Directory containing model folders
metrics_dir = "/large_storage/hsulab/kzhu/DART_work_dir/task_2_footprinting/outputs/evals/likelihoods"

# Initialize data storage for model names and accuracies
model_names = []
accuracies = []

for model_metrics in os.listdir(metrics_dir):
    metrics_file = os.path.join(metrics_dir, model_metrics)

    if os.path.isfile(metrics_file):
        try:
            df = pd.read_csv(metrics_file, sep='\t')
            accuracies.append(df["Accuracy"].mean())
        except:
            continue
        model_names.append(model_metrics.split(".")[0])

# Separate evo and non-evo models
evo_models = [(name, acc) for name, acc in zip(model_names, accuracies) if "evo" in name.lower()]
non_evo_models = [(name, acc) for name, acc in zip(model_names, accuracies) if "evo" not in name.lower()]

# Sort each group by accuracy
evo_models.sort(key=lambda x: x[1], reverse=True)
non_evo_models.sort(key=lambda x: x[1], reverse=True)

# Combine back together
model_names = [m[0] for m in evo_models + non_evo_models]
accuracies = [m[1] for m in evo_models + non_evo_models]

# Calculate positions for optimal spacing
bar_width = 0.25
gap_width = 0.05
x_pos = np.arange(len(model_names)) * (bar_width + gap_width)

# Define colors based on model names
colors = ['#FF8081' if 'evo' in name.lower() else '#B7BDC8' for name in model_names]

plt.figure(figsize=(8, 10))
bars = plt.bar(x_pos, accuracies, color=colors,
               width=bar_width,
               align='center')

plt.xticks(x_pos, model_names, rotation=45, ha='right')
plt.xlabel("Model", fontsize=14, labelpad=10)
plt.ylabel("Accuracy", fontsize=14, labelpad=10)
plt.title("DART Task 2 Zero-Shot LLs Accuracy Comparison", fontsize=16, weight='bold', pad=15)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add accuracy values on top of each bar
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, 
             f'{acc:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig("DART_t2_zeroshot_lls_plot.png", dpi=300)
plt.show()
