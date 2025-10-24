import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load the preprocessed data
X_train = pd.read_csv('X_train_scaled.csv')
X_test = pd.read_csv('X_test_scaled.csv')
y_train = pd.read_csv('y_train.csv')['diagnosis'].values
y_test = pd.read_csv('y_test.csv')['diagnosis'].values

# Handle NaN values
X_train = X_train.dropna(axis=1, how='all').fillna(0)
X_test = X_test.dropna(axis=1, how='all').fillna(0)

# Train Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Get the linear combination (z) and probabilities for test samples
z_values = model.decision_function(X_test)  # This gives us z = b0 + b1*x1 + ... + bn*xn
probabilities = model.predict_proba(X_test)[:, 1]  # This gives us sigmoid(z)

print("SIGMOID FUNCTION DEMONSTRATION")
print("=" * 50)
print("Sample of test data showing the relationship between z and sigmoid(z):")
print()

# Show first 10 test samples
print(f"{'Sample':<8} {'z (linear)':<12} {'sigmoid(z)':<12} {'Actual':<8} {'Predicted':<10}")
print("-" * 60)

for i in range(min(10, len(z_values))):
    actual = "Malignant" if y_test[i] == 1 else "Benign"
    predicted = "Malignant" if probabilities[i] >= 0.5 else "Benign"
    print(f"{i+1:<8} {z_values[i]:<12.4f} {probabilities[i]:<12.4f} {actual:<8} {predicted:<10}")

print()
print("KEY OBSERVATIONS:")
print("- When z is large positive (>2), sigmoid(z) approaches 1 (high malignant probability)")
print("- When z is large negative (<-2), sigmoid(z) approaches 0 (low malignant probability)")
print("- When z is around 0, sigmoid(z) is around 0.5 (uncertain)")
print("- The sigmoid function smoothly converts any z value to a probability between 0 and 1")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Sigmoid function with actual data points
z_range = np.linspace(-6, 6, 100)
sigmoid_range = 1 / (1 + np.exp(-z_range))

ax1.plot(z_range, sigmoid_range, 'b-', linewidth=2, label='Sigmoid Function')
ax1.scatter(z_values, probabilities, c=y_test, cmap='RdYlBu', alpha=0.7, s=50)
ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold = 0.5')
ax1.axvline(x=0, color='g', linestyle='--', alpha=0.7, label='z = 0')
ax1.set_xlabel('z (linear combination)')
ax1.set_ylabel('sigmoid(z) (probability)')
ax1.set_title('Sigmoid Function with Actual Data Points')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Color bar for the scatter plot
cbar = plt.colorbar(ax1.collections[0], ax=ax1)
cbar.set_label('Actual Class (0=Benign, 1=Malignant)')

# Plot 2: Distribution of z values by class
ax2.hist(z_values[y_test == 0], bins=20, alpha=0.7, label='Benign (z values)', color='blue')
ax2.hist(z_values[y_test == 1], bins=20, alpha=0.7, label='Malignant (z values)', color='red')
ax2.axvline(x=0, color='g', linestyle='--', alpha=0.7, label='z = 0 (sigmoid = 0.5)')
ax2.set_xlabel('z (linear combination)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of z values by Actual Class')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sigmoid_demonstration.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved as 'sigmoid_demonstration.png'")
print("\nThis demonstrates how the sigmoid function converts linear combinations")
print("of features into probabilities, with clear separation between classes.")
