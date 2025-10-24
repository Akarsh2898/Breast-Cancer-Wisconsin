import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading Breast Cancer Dataset...")
df = pd.read_csv('data.csv')

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nTarget variable distribution:")
print(df['diagnosis'].value_counts())

# Check for missing values
print(f"\nMissing values per column:")
print(df.isnull().sum())

# Remove the 'id' column as it's not useful for classification
df = df.drop('id', axis=1)

# Encode the target variable: M (malignant) = 1, B (benign) = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

print(f"\nTarget variable after encoding:")
print(df['diagnosis'].value_counts())

# Separate features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Ensures both classes are represented proportionally
)

print(f"\nTrain set shape: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")

print(f"\nTraining set target distribution:")
print(y_train.value_counts())

print(f"\nTest set target distribution:")
print(y_test.value_counts())

# Standardize the features
print("\nStandardizing features...")
scaler = StandardScaler()

# Fit the scaler on training data only
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Feature standardization completed!")
print(f"\nTraining set statistics after scaling:")
print(f"Mean: {X_train_scaled.mean().mean():.6f}")
print(f"Std: {X_train_scaled.std().mean():.6f}")

print(f"\nTest set statistics after scaling:")
print(f"Mean: {X_test_scaled.mean().mean():.6f}")
print(f"Std: {X_test_scaled.std().mean():.6f}")

# Display some statistics about the scaled features
print(f"\nScaled training features summary:")
print(X_train_scaled.describe())

# Save the processed data
print("\nSaving processed data...")
X_train_scaled.to_csv('X_train_scaled.csv', index=False)
X_test_scaled.to_csv('X_test_scaled.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Processed data saved to:")
print("- X_train_scaled.csv")
print("- X_test_scaled.csv") 
print("- y_train.csv")
print("- y_test.csv")

# Create a summary of the preprocessing
print("\n" + "="*50)
print("PREPROCESSING SUMMARY")
print("="*50)
print(f"Original dataset shape: {df.shape}")
print(f"Features used: {X.shape[1]}")
print(f"Training samples: {X_train.shape[0]} ({X_train.shape[0]/df.shape[0]*100:.1f}%)")
print(f"Test samples: {X_test.shape[0]} ({X_test.shape[0]/df.shape[0]*100:.1f}%)")
print(f"Target classes: {df['diagnosis'].nunique()} (0=Benign, 1=Malignant)")
print(f"Features standardized: Yes")
print(f"Random state: 42 (for reproducibility)")
print("="*50)
