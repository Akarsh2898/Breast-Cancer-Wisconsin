import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data
print("Loading preprocessed data...")
X_train = pd.read_csv('X_train_scaled.csv')
X_test = pd.read_csv('X_test_scaled.csv')
y_train = pd.read_csv('y_train.csv')['diagnosis'].values
y_test = pd.read_csv('y_test.csv')['diagnosis'].values

# Handle NaN values
X_train = X_train.dropna(axis=1, how='all').fillna(0)
X_test = X_test.dropna(axis=1, how='all').fillna(0)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Train Logistic Regression model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calculate precision, recall, F1-score
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Specificity
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n" + "="*60)
print("LOGISTIC REGRESSION EVALUATION RESULTS")
print("="*60)

print(f"\nCONFUSION MATRIX:")
print(f"                 Predicted")
print(f"Actual    Benign  Malignant")
print(f"Benign    {tn:6d}  {fp:6d}")
print(f"Malignant {fn:6d}  {tp:6d}")

print(f"\nPERFORMANCE METRICS:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1_score:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"ROC-AUC:   {auc_roc:.4f}")

print(f"\nDETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# Create a simple visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Malignant'], 
            yticklabels=['Benign', 'Malignant'],
            ax=ax1)
ax1.set_title('Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend(loc="lower right")

plt.tight_layout()
plt.savefig('logistic_regression_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved as 'logistic_regression_evaluation.png'")
print("="*60)
