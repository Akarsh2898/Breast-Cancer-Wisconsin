import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           roc_auc_score, roc_curve, precision_recall_curve, 
                           precision_score, recall_score, f1_score)
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

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "="*80)
print("UNDERSTANDING THE SIGMOID FUNCTION IN LOGISTIC REGRESSION")
print("="*80)

print("""
The sigmoid function is the core mathematical function used in logistic regression to convert 
linear combinations of features into probabilities between 0 and 1.

Mathematical Formula:
sigmoid(z) = 1 / (1 + e^(-z))

Where:
- z = b0 + b1*x1 + b2*x2 + ... + bn*xn (linear combination of features)
- e = Euler's number (~2.718)
- sigmoid(z) = probability of the positive class (malignant)

Key Properties:
1. Output Range: (0, 1) - perfect for probabilities
2. S-shaped curve: smooth transition between 0 and 1
3. Symmetric around 0.5 when z = 0
4. As z approaches +infinity, sigmoid(z) approaches 1
5. As z approaches -infinity, sigmoid(z) approaches 0

In our breast cancer model:
- z = linear combination of 30 standardized features
- sigmoid(z) = probability that a tumor is malignant
- Default threshold = 0.5 (if sigmoid(z) >= 0.5, predict malignant)
""")

# Demonstrate sigmoid function
z_values = np.linspace(-6, 6, 100)
sigmoid_values = 1 / (1 + np.exp(-z_values))

# Create comprehensive threshold analysis
print("\n" + "="*80)
print("THRESHOLD TUNING ANALYSIS")
print("="*80)

# Test different thresholds
thresholds = np.arange(0.1, 0.95, 0.05)
results = []

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_thresh)
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    # Confusion matrix components
    cm = confusion_matrix(y_test, y_pred_thresh)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # Same as recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'fpr': fpr,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    })

results_df = pd.DataFrame(results)

# Find optimal thresholds for different criteria
print("\nOPTIMAL THRESHOLDS FOR DIFFERENT CRITERIA:")
print("-" * 50)

# Best F1-score
best_f1_idx = results_df['f1_score'].idxmax()
best_f1_thresh = results_df.loc[best_f1_idx, 'threshold']
print(f"Best F1-Score: {results_df.loc[best_f1_idx, 'f1_score']:.4f} at threshold {best_f1_thresh:.2f}")

# Best Accuracy
best_acc_idx = results_df['accuracy'].idxmax()
best_acc_thresh = results_df.loc[best_acc_idx, 'threshold']
print(f"Best Accuracy: {results_df.loc[best_acc_idx, 'accuracy']:.4f} at threshold {best_acc_thresh:.2f}")

# Best Precision
best_prec_idx = results_df['precision'].idxmax()
best_prec_thresh = results_df.loc[best_prec_idx, 'threshold']
print(f"Best Precision: {results_df.loc[best_prec_idx, 'precision']:.4f} at threshold {best_prec_thresh:.2f}")

# Best Recall (Sensitivity)
best_recall_idx = results_df['recall'].idxmax()
best_recall_thresh = results_df.loc[best_recall_idx, 'threshold']
print(f"Best Recall: {results_df.loc[best_recall_idx, 'recall']:.4f} at threshold {best_recall_idx:.2f}")

# Balanced (where precision = recall)
precision_recall_diff = np.abs(results_df['precision'] - results_df['recall'])
balanced_idx = precision_recall_diff.idxmin()
balanced_thresh = results_df.loc[balanced_idx, 'threshold']
print(f"Most Balanced (Precision ~ Recall): threshold {balanced_thresh:.2f}")

print("\n" + "="*80)
print("DETAILED THRESHOLD ANALYSIS")
print("="*80)

# Display detailed results for key thresholds
key_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, best_f1_thresh]
print(f"\n{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Specificity':<12}")
print("-" * 70)

for thresh in key_thresholds:
    if thresh in results_df['threshold'].values:
        row = results_df[results_df['threshold'] == thresh].iloc[0]
        print(f"{thresh:<10.2f} {row['accuracy']:<10.4f} {row['precision']:<10.4f} {row['recall']:<10.4f} {row['f1_score']:<10.4f} {row['specificity']:<12.4f}")

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Sigmoid Function Visualization
axes[0, 0].plot(z_values, sigmoid_values, 'b-', linewidth=2, label='Sigmoid Function')
axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Default Threshold (0.5)')
axes[0, 0].axvline(x=0, color='g', linestyle='--', alpha=0.7, label='z = 0')
axes[0, 0].set_xlabel('z (linear combination)')
axes[0, 0].set_ylabel('sigmoid(z) (probability)')
axes[0, 0].set_title('Sigmoid Function: sigmoid(z) = 1/(1+e^(-z))')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Threshold vs Performance Metrics
axes[0, 1].plot(results_df['threshold'], results_df['accuracy'], 'b-', label='Accuracy', linewidth=2)
axes[0, 1].plot(results_df['threshold'], results_df['precision'], 'r-', label='Precision', linewidth=2)
axes[0, 1].plot(results_df['threshold'], results_df['recall'], 'g-', label='Recall', linewidth=2)
axes[0, 1].plot(results_df['threshold'], results_df['f1_score'], 'm-', label='F1-Score', linewidth=2)
axes[0, 1].axvline(x=best_f1_thresh, color='k', linestyle='--', alpha=0.7, label=f'Best F1: {best_f1_thresh:.2f}')
axes[0, 1].set_xlabel('Threshold')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('Performance Metrics vs Threshold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. ROC Curve with different thresholds
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)
axes[0, 2].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Classifier')

# Mark some threshold points on ROC curve
threshold_points = [0.3, 0.5, 0.7]
for thresh in threshold_points:
    if thresh in roc_thresholds:
        idx = np.where(roc_thresholds == thresh)[0][0]
        axes[0, 2].plot(fpr[idx], tpr[idx], 'ro', markersize=8, label=f'Threshold {thresh}')

axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].set_title('ROC Curve with Threshold Points')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Precision-Recall Curve
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
axes[1, 0].plot(recall_curve, precision_curve, 'b-', linewidth=2, label='Precision-Recall Curve')
axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Baseline (0.5)')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision-Recall Curve')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Confusion Matrix Heatmap for different thresholds
thresholds_to_show = [0.3, 0.5, 0.7]
for i, thresh in enumerate(thresholds_to_show):
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_thresh)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], 
                yticklabels=['Benign', 'Malignant'],
                ax=axes[1, 1] if i == 1 else axes[1, 2])
    
    if i == 1:  # Use middle subplot for 0.5 threshold
        axes[1, 1].set_title(f'Confusion Matrix (Threshold = {thresh})')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
    elif i == 0:  # Use right subplot for 0.3 threshold
        axes[1, 2].set_title(f'Confusion Matrix (Threshold = {thresh})')
        axes[1, 2].set_xlabel('Predicted')
        axes[1, 2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('threshold_tuning_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Medical context analysis
print("\n" + "="*80)
print("MEDICAL CONTEXT ANALYSIS")
print("="*80)

print("""
In medical diagnosis, the choice of threshold depends on the clinical context:

1. CONSERVATIVE APPROACH (Lower Threshold, e.g., 0.3):
   - Higher Recall (Sensitivity) - catches more malignant cases
   - Lower Precision - more false positives
   - Better for screening where missing cancer is worse than false alarms
   - More patients referred for further testing

2. BALANCED APPROACH (Threshold 0.5):
   - Equal weight to precision and recall
   - Good general-purpose threshold
   - Current default in our model

3. STRICT APPROACH (Higher Threshold, e.g., 0.7):
   - Higher Precision - fewer false positives
   - Lower Recall - might miss some malignant cases
   - Better when false positives are costly or harmful
   - Fewer unnecessary procedures

RECOMMENDATION FOR BREAST CANCER:
- Use threshold around 0.4-0.5 for screening
- Use threshold around 0.6-0.7 for diagnostic confirmation
- Consider patient risk factors and family history
""")

# Save detailed results
results_df.to_csv('threshold_analysis_results.csv', index=False)
print(f"\nDetailed threshold analysis saved to 'threshold_analysis_results.csv'")
print(f"Visualization saved to 'threshold_tuning_analysis.png'")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"• Best F1-Score threshold: {best_f1_thresh:.2f}")
print(f"• Best Accuracy threshold: {best_acc_thresh:.2f}")
print(f"• Most balanced threshold: {balanced_thresh:.2f}")
print(f"• Current model AUC: {auc_score:.4f}")
print("• Threshold tuning allows optimization for specific clinical needs")
print("• Lower thresholds = higher sensitivity (catch more cancers)")
print("• Higher thresholds = higher specificity (fewer false alarms)")
print("="*80)
