import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Data
mlenet_accuracy = [70, 71, 70, 71, 70, 70, 57, 66, 69, 69, 58, 69]
vithar_accuracy = [71, 75, 70, 69, 68, 72, 61, 78, 74, 73, 61, 77]

# Perform t-test
t_stat, p_value = stats.ttest_ind(mlenet_accuracy, vithar_accuracy)

# Print t-statistic and p-value
print("T-Statistic:", t_stat)
print("P-Value:", p_value)

# Visualization with Boxplot
data = {'Model': ['M-LeNet'] * len(mlenet_accuracy) + ['ViT-HAR'] * len(vithar_accuracy),
        'Accuracy': mlenet_accuracy + vithar_accuracy}

plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='Accuracy', data=data)
plt.title('Comparison of M-LeNet and ViT-HAR Accuracies')
plt.xlabel('Model')
plt.ylabel('Accuracy(%)')
plt.show()
