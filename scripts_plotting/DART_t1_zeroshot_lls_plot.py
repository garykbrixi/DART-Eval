import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Directory containing model folders
metrics_dir = "/large_storage/hsulab/kzhu/DART_work_dir/task_1_ccre/zero_shot_outputs/likelihoods"

# Initialize data storage for model names and accuracies
model_names = []
accuracies = []

# Loop over each model folder
for model_folder in os.listdir(metrics_dir):
    folder_path = os.path.join(metrics_dir, model_folder)
    metrics_file = os.path.join(folder_path, "metrics.json")
    
    if os.path.isfile(metrics_file):
        # Load accuracy from metrics.json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            if "acc" in metrics:
                model_names.append(model_folder)
                accuracies.append(metrics["acc"])

# Separate and sort evo and non-evo models
evo_models = [(name, acc) for name, acc in zip(model_names, accuracies) if "evo" in name.lower()]
non_evo_models = [(name, acc) for name, acc in zip(model_names, accuracies) if "evo" not in name.lower()]

# Sort each group by accuracy
evo_models.sort(key=lambda x: x[1], reverse=True)
non_evo_models.sort(key=lambda x: x[1], reverse=True)

# Combine back together
model_names = [m[0] for m in evo_models + non_evo_models]
accuracies = [m[1] for m in evo_models + non_evo_models]
colors = ['#FF8081' if 'evo' in name.lower() else '#B7BDC8' for name in model_names]

# Calculate positions for optimal spacing
bar_width = 0.2  # Reduced bar width
gap_width = 0.05  # Gap is still 1/5 of bar width
x_pos = np.arange(len(model_names)) * (bar_width + gap_width)

plt.figure(figsize=(8, 10))  # Changed from (15, 8) to (8, 10) for tall/skinny look
bars = plt.bar(x_pos, accuracies, color=colors,
               width=bar_width,
               align='center')

# Adjust x-axis ticks with rotated labels
plt.xticks(x_pos, model_names, rotation=45, ha='right')

plt.xlabel("Model", fontsize=14, labelpad=10)
plt.ylabel("Accuracy", fontsize=14, labelpad=10)
plt.title("DART Task 1 Zero-Shot LLs Accuracy Comparison", fontsize=16, weight='bold', pad=15)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add accuracy values on top of each bar
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{acc:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')

# Adjust layout to prevent label cutoff
plt.tight_layout()

plt.savefig("DART_t1_zeroshot_lls_plot.png", dpi=300)
plt.show()
