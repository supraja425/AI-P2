import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('File Path')

# Clean column names (if there are spaces or unwanted characters)
data.columns = data.columns.str.strip()

# Check the column names
print("Columns in the dataset:", data.columns)

# Handle missing and infinite values
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Handle categorical columns like 'Label'
data['Label'] = data['Label'].fillna(data['Label'].mode()[0])  # Replace missing with the most frequent category

# Drop columns that are not needed (adjust as necessary)
columns_to_drop = ['Destination Port', 'IPsrc', 'IPdst', 'Timestamp']  # Adjust column names as necessary
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1)

# Feature Extraction and Label Separation
X = data.drop('Label', axis=1, errors='ignore')  # Features (everything except 'Label')
y = data['Label'] if 'Label' in data.columns else None  # Labels (target column)

if y is None:
    print("No 'Label' column found. Please ensure you have a target column for classification.")
else:
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training: Using RandomForestClassifier for simplicity
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predicting on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluation of the model
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix (Optional, but helps in evaluating classification accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Visualization: Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'MALICIOUS'], yticklabels=['BENIGN', 'MALICIOUS'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Feature Importance Plot
    feature_importance = model.feature_importances_
    features = X.columns
    sorted_idx = np.argsort(feature_importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(features[sorted_idx], feature_importance[sorted_idx])
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    # Visualizing the Malicious vs. Benign Traffic Distribution
    label_counts = data['Label'].value_counts()
    plt.figure(figsize=(6, 4))
    label_counts.plot(kind='bar', color=['blue', 'orange'])
    plt.title('Distribution of Benign vs Malicious Traffic')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Benign', 'Malicious'], rotation=0)
    plt.show()
