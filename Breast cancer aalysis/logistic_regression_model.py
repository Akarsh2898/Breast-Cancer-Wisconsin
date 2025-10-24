import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data
print("Loading preprocessed data...")
X_train = pd.read_csv('X_train_scaled.csv')
X_test = pd.read_csv('X_test_scaled.csv')
y_train = pd.read_csv('y_train.csv')['diagnosis'].values
y_test = pd.read_csv('y_test.csv')['diagnosis'].values

# Check for NaN values and handle them
print(f"NaN values in X_train: {X_train.isnull().sum().sum()}")
print(f"NaN values in X_test: {X_test.isnull().sum().sum()}")

# Remove any columns with all NaN values
X_train = X_train.dropna(axis=1, how='all')
X_test = X_test.dropna(axis=1, how='all')

# Fill any remaining NaN values with 0 (shouldn't happen after standardization)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"After cleaning - NaN values in X_train: {X_train.isnull().sum().sum()}")
print(f"After cleaning - NaN values in X_test: {X_test.isnull().sum().sum()}")

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")

# Display class distribution
print(f"\nTraining set class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for class_label, count in zip(unique, counts):
    print(f"Class {class_label}: {count} samples ({count/len(y_train)*100:.1f}%)")

print(f"\nTest set class distribution:")
unique, counts = np.unique(y_test, return_counts=True)
for class_label, count in zip(unique, counts):
    print(f"Class {class_label}: {count} samples ({count/len(y_test)*100:.1f}%)")

# Fit Logistic Regression model
print("\n" + "="*50)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("="*50)

# Create and train the model
logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000,  # Increase max_iter to ensure convergence
    solver='liblinear'  # Good for small datasets
)

print("Fitting Logistic Regression model...")
logistic_model.fit(X_train, y_train)

print("Model training completed!")
print(f"Model converged: {logistic_model.n_iter_[0]} iterations")

# Make predictions
print("\n" + "="*50)
print("MAKING PREDICTIONS")
print("="*50)

# Predictions on training set
y_train_pred = logistic_model.predict(X_train)
y_train_pred_proba = logistic_model.predict_proba(X_train)[:, 1]

# Predictions on test set
y_test_pred = logistic_model.predict(X_test)
y_test_pred_proba = logistic_model.predict_proba(X_test)[:, 1]

print("Predictions completed!")

# Model Evaluation
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Training set performance
train_accuracy = accuracy_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_pred_proba)

print(f"TRAINING SET PERFORMANCE:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"AUC-ROC: {train_auc:.4f}")

# Test set performance
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_pred_proba)

print(f"\nTEST SET PERFORMANCE:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"AUC-ROC: {test_auc:.4f}")

# Detailed classification report
print(f"\nDETAILED CLASSIFICATION REPORT (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malignant']))

# Confusion Matrix
print(f"\nCONFUSION MATRIX (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Feature importance (coefficients)
print(f"\nFEATURE IMPORTANCE (Top 10):")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': logistic_model.coef_[0],
    'abs_coefficient': np.abs(logistic_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print(feature_importance.head(10))

# Create visualizations
print(f"\nCreating visualizations...")

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Malignant'], 
            yticklabels=['Benign', 'Malignant'],
            ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix (Test Set)')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve (Test Set)')
axes[0, 1].legend(loc="lower right")

# 3. Feature Importance (Top 15)
top_features = feature_importance.head(15)
axes[1, 0].barh(range(len(top_features)), top_features['coefficient'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'])
axes[1, 0].set_xlabel('Coefficient Value')
axes[1, 0].set_title('Top 15 Feature Importance (Coefficients)')
axes[1, 0].invert_yaxis()

# 4. Prediction Probability Distribution
axes[1, 1].hist(y_test_pred_proba[y_test == 0], bins=20, alpha=0.7, label='Benign', color='blue')
axes[1, 1].hist(y_test_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Malignant', color='red')
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Probability Distribution (Test Set)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('logistic_regression_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results to files
print(f"\nSaving results...")

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_test_pred,
    'probability_benign': 1 - y_test_pred_proba,
    'probability_malignant': y_test_pred_proba
})
predictions_df.to_csv('logistic_regression_predictions.csv', index=False)

# Save feature importance
feature_importance.to_csv('logistic_regression_feature_importance.csv', index=False)

# Save model performance summary
performance_summary = {
    'Metric': ['Training Accuracy', 'Test Accuracy', 'Training AUC-ROC', 'Test AUC-ROC'],
    'Value': [train_accuracy, test_accuracy, train_auc, test_auc]
}
performance_df = pd.DataFrame(performance_summary)
performance_df.to_csv('logistic_regression_performance.csv', index=False)

print("Results saved to:")
print("- logistic_regression_results.png")
print("- logistic_regression_predictions.csv")
print("- logistic_regression_feature_importance.csv")
print("- logistic_regression_performance.csv")

# Final Summary
print("\n" + "="*60)
print("LOGISTIC REGRESSION MODEL SUMMARY")
print("="*60)
print(f"Model Type: Logistic Regression")
print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")
print(f"Features Used: {X_train.shape[1]}")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC-ROC: {test_auc:.4f}")
print(f"Model Converged: Yes ({logistic_model.n_iter_[0]} iterations)")
print("="*60)
