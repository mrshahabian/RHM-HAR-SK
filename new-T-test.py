import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
# import seaborn as sns

mlenet_accuracy = [70, 71, 70, 71, 70, 70, 57, 66, 69, 69, 58, 69]
vithar_accuracy = [71, 75, 70, 69, 68, 72, 61, 78, 74, 73, 61, 77]

# Normality tests
def check_normality(data, model_name):
    # Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins='auto', alpha=0.7, label=model_name)
    plt.title(f'Histogram for {model_name}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Q-Q Plot
    plt.figure(figsize=(8, 4))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot for {model_name}')
    plt.show()

    # Shapiro-Wilk Test
    shapiro_test = stats.shapiro(data)
    print(f"Shapiro-Wilk Test for {model_name}: W={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")

# Check normality for both models
check_normality(mlenet_accuracy, 'MLENet')
check_normality(vithar_accuracy, 'ViTHar')

# Variance test
levene_test = stats.levene(mlenet_accuracy, vithar_accuracy)
print(f"Levene's Test for equal variances: W={levene_test.statistic}, p-value={levene_test.pvalue}")

# Perform paired T-test
t_stat, p_value = stats.ttest_rel(mlenet_accuracy, vithar_accuracy)

print(f"Paired T-test: T-statistic={t_stat}, p-value={p_value}")
