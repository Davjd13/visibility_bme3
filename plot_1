import matplotlib.pyplot as plt
import numpy as np

# Accuracy data
knn_data = np.array([
    [0.6256, 0.5933, 0.5678, 0.5878], #normal
    [0.5622, 0.5833, 0.7022, 0.6489],
    [0.582, 0.6379, 0.6042, 0.5878],
    [0.6524, 0.6563, 0.6624, 0.6989]
    # [0.3167, 0.325,  0.35,   0.2167], #prob2
    # [0.5354, 0.5538, 0.6452, 0.4667],
    # [0.4167, 0.4,    0.2917, 0.3833],
    # [0.4957, 0.4994, 0.6262, 0.2167]
])

svm_data = np.array([
    [0.7011, 0.6567, 0.7289, 0.6789], #normal
    [0.6767, 0.6911, 0.7378, 0.7178],
    [0.6498, 0.6821, 0.6608, 0.6789],
    [0.6784, 0.6955, 0.73, 0.7089]
    # [0.5583, 0.6917, 0.6,    0.7167], #prob2
    # [0.5777, 0.5782, 0.6167, 0.6167],
    # [0.6083, 0.5833, 0.5583, 0.6167],
    # [0.5658, 0.6167, 0.6476, 0.7167]
])

# Multiply accuracy by 100
knn_data *= 100
svm_data *= 100

# Plot settings
x_axis = [512, 1024, 2048, 4096]
test_cases = ["Test case 1", "Test case 2", "Test case 3", "Test case 4"]
colors = ['red', 'blue', 'green', 'orange']
markers = ['o', 's', '^', 'D']

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot for KNN
for i, test_case in enumerate(test_cases):
    axs[0].plot(x_axis, knn_data[i], label=test_case, color=colors[i], marker=markers[i], linestyle='--', linewidth=7, markersize=15)

axs[0].set_title('KNN', fontsize=35)
axs[0].set_xlabel('Nodes', fontsize=35)
axs[0].set_ylabel('Accuracy (%)', fontsize=35)
axs[0].legend(fontsize=22)
axs[0].grid(True)
axs[0].set_ylim(55, 80)
axs[0].set_xticks(x_axis)
axs[0].tick_params(axis='both', which='major', labelsize=25)

# Plot for SVM
for i, test_case in enumerate(test_cases):
    axs[1].plot(x_axis, svm_data[i], label=test_case, color=colors[i], marker=markers[i], linestyle='--', linewidth=7, markersize=15)

axs[1].set_title('SVM', fontsize=35)
axs[1].set_xlabel('Nodes', fontsize=35)
axs[1].set_ylabel('Accuracy (%)', fontsize=35)
axs[1].legend(fontsize=25)
axs[1].grid(True)
axs[1].set_ylim(55, 80)
axs[1].set_xticks(x_axis)
axs[1].tick_params(axis='both', which='major', labelsize=25)

plt.tight_layout()
plt.show()
