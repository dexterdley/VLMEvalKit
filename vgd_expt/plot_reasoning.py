import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

alpha_values = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.5])

qwen_2b_scores = {}
qwen_4b_scores = {}
qwen_8b_scores = {}

qwen_2b_scores[42] = np.array([0.321, 0.343, 0.349, 0.364, 0.359, 0.349])
qwen_4b_scores[42] = np.array([0.373, 0.374, 0.398, 0.395, 0.388, 0.400])
qwen_8b_scores[42] = np.array([0.391, 0.417, 0.429, 0.444, 0.450, 0.467])

qwen_2b_scores[55] = np.array([0.317, 0.347, 0.346, 0.349, 0.345, 0.352])
qwen_4b_scores[55] = np.array([0.371, 0.375, 0.391, 0.393, 0.397, 0.395])
qwen_8b_scores[55] = np.array([0.399, 0.426, 0.433, 0.433, 0.451, 0.457])

qwen_2b_scores[69] = np.array([0.315, 0.340, 0.345, 0.353, 0.351, 0.359])
qwen_4b_scores[69] = np.array([0.368, 0.383, 0.385, 0.400, 0.399, 0.403])
qwen_8b_scores[69] = np.array([0.386, 0.416, 0.429, 0.446, 0.455, 0.469])

# Calculate mean and std across seeds
qwen_2b_all = np.array([qwen_2b_scores[42], qwen_2b_scores[55], qwen_2b_scores[69]])
qwen_4b_all = np.array([qwen_4b_scores[42], qwen_4b_scores[55], qwen_4b_scores[69]])
qwen_8b_all = np.array([qwen_8b_scores[42], qwen_8b_scores[55], qwen_8b_scores[69]])

qwen_2b_mean = np.mean(qwen_2b_all, axis=0)
qwen_2b_std = np.std(qwen_2b_all, axis=0)

qwen_4b_mean = np.mean(qwen_4b_all, axis=0)
qwen_4b_std = np.std(qwen_4b_all, axis=0)

qwen_8b_mean = np.mean(qwen_8b_all, axis=0)
qwen_8b_std = np.std(qwen_8b_all, axis=0)

# 2. Plot Setup
plt.figure(figsize=(8, 5))

# Helper function to plot Data + Trend + Shaded Std
def plot_data_with_arrow(x, y_mean, y_std, label, color):
    # A. Plot the shaded standard deviation region
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)
    
    # B. Plot the mean line with markers
    plt.plot(x, y_mean, marker="o", label=label, color=color, linewidth=2)

    # C. Calculate Linear Trend Coordinates
    slope, intercept = np.polyfit(x, y_mean, 1)

    x_start, x_end = x[0], x[-1]
    y_start = slope * x_start + intercept
    y_end = slope * x_end + intercept

    # D. Draw the Dashed Arrow
    plt.annotate(
        "",
        xy=(x_end, y_end),
        xytext=(x_start, y_start),
        arrowprops=dict(
            arrowstyle="->",
            linestyle="--",
            color="gray",
            linewidth=1.5,
            alpha=0.6,
            shrinkA=0, shrinkB=0
        )
    )

# Draw Plots
plot_data_with_arrow(alpha_values, qwen_2b_mean, qwen_2b_std, "Qwen3-VL-2B", 'b')
plot_data_with_arrow(alpha_values, qwen_4b_mean, qwen_4b_std, "Qwen3-VL-4B", 'g')
plot_data_with_arrow(alpha_values, qwen_8b_mean, qwen_8b_std, "Qwen3-VL-8B", 'r')

# 4. Styling
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xlabel("Visual Guidance (Alpha)", fontsize=18)
plt.ylabel("Reasoning Accuracy", fontsize=18)
#plt.xlim(1, 2.5)
plt.yticks([0.32, 0.36, 0.40, 0.44], fontsize=12) # Optional: Adjust tick size for readability
plt.xticks(fontsize=12)
plt.legend(fontsize=10, loc="upper left")
plt.grid(True, alpha=0.2) # Lighter grid

plt.tight_layout()
plt.savefig(f"./reasoning_curves.pdf", dpi=100, bbox_inches='tight')
plt.show()
print("Reasoning plot saved")