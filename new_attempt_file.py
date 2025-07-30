# parkinsons_prediction.ipynb
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading and Initial Exploration ---

print("--- Data Loading and Initial Exploration ---")

# Define the path to your dataset.
# IMPORTANT: In a real Jupyter environment, you might upload the CSV file
# or ensure it's in the same directory as this notebook.
# For demonstration, we'll assume 'parkinsons.data' is available.
try:
    df = pd.read_csv('parkinsons.data')
    print("Dataset loaded successfully.")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Information:")
    df.info()
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())
except FileNotFoundError:
    print("Error: 'parkinsons.data' not found.")
    print("Please ensure the dataset CSV file is in the same directory as this notebook,")
    print("or provide the correct path to the file.")
    # Create a dummy DataFrame for demonstration if file not found
    print("\nCreating a dummy dataset for demonstration purposes...")
    data = {
        'name': [f'Sub{i}' for i in range(10)],
        'MDVP:Fo(Hz)': np.random.rand(10) * 100 + 100,
        'MDVP:Fhi(Hz)': np.random.rand(10) * 200 + 200,
        'MDVP:Flo(Hz)': np.random.rand(10) * 50 + 50,
        'MDVP:Jitter(%)': np.random.rand(10) * 0.01,
        'MDVP:Jitter(Abs)': np.random.rand(10) * 0.001,
        'MDVP:RAP': np.random.rand(10) * 0.005,
        'MDVP:PPQ': np.random.rand(10) * 0.005,
        'Jitter:DDP': np.random.rand(10) * 0.015,
        'MDVP:Shimmer': np.random.rand(10) * 0.05,
        'MDVP:Shimmer(dB)': np.random.rand(10) * 0.5,
        'Shimmer:APQ3': np.random.rand(10) * 0.03,
        'Shimmer:APQ5': np.random.rand(10) * 0.03,
        'MDVP:APQ': np.random.rand(10) * 0.04,
        'Shimmer:DDA': np.random.rand(10) * 0.09,
        'NHR': np.random.rand(10) * 0.05,
        'HNR': np.random.rand(10) * 20 + 10,
        'RPDE': np.random.rand(10) * 0.5 + 0.5,
        'DFA': np.random.rand(10) * 0.5 + 0.5,
        'spread1': np.random.rand(10) * -5 - 1,
        'spread2': np.random.rand(10) * 0.2,
        'D2': np.random.rand(10) * 3 + 1,
        'PPE': np.random.rand(10) * 0.2 + 0.1,
        'status': np.random.randint(0, 2, 10) # 0 for healthy, 1 for Parkinson's
    }
    df = pd.DataFrame(data)
    print("\nDummy dataset created. Please replace with actual 'parkinsons.data' for real results.")


# --- 2. Data Preprocessing and Feature Scaling ---

print("\n--- Data Preprocessing and Feature Scaling ---")

# Drop the 'name' column as it's an identifier and not a feature for prediction
df = df.drop('name', axis=1)

# Separate features (X) and target (y)
X = df.drop('status', axis=1) # All columns except 'status' are features
y = df['status']             # 'status' is the target variable (0: healthy, 1: Parkinson's)

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Split the dataset into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing
# random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training features (X_train) shape: {X_train.shape}")
print(f"Testing features (X_test) shape: {X_test.shape}")
print(f"Training target (y_train) shape: {y_train.shape}")
print(f"Testing target (y_test) shape: {y_test.shape}")

# Initialize the StandardScaler
# StandardScaler standardizes features by removing the mean and scaling to unit variance.
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled successfully using StandardScaler.")


# --- 3. Logistic Regression Model Training ---

print("\n--- Logistic Regression Model Training ---")

# Initialize the Logistic Regression model
# random_state for reproducibility
model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' solver is good for small datasets

# Train the model using the scaled training data
model.fit(X_train_scaled, y_train)

print("Logistic Regression model trained successfully.")


# --- 4. Model Evaluation ---

print("\n--- Model Evaluation ---")

# Predict on the scaled test data
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probabilities for the positive class (1: Parkinson's)

# Calculate Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")

# Display Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Healthy (0)', 'Predicted PD (1)'],
            yticklabels=['Actual Healthy (0)', 'Actual PD (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Display Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot ROC Curve and calculate AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"ROC AUC Score: {roc_auc:.4f}")


# --- 5. Dimensionality Reduction with PCA for Visualization ---

print("\n--- Dimensionality Reduction with PCA for Visualization ---")

# Initialize PCA to reduce to 2 components for visualization
pca = PCA(n_components=2)

# Fit PCA on the scaled training data and transform both training and testing data
# For visualization, we'll transform the entire scaled dataset (X_scaled)
# to see the overall distribution in 2D.
X_scaled_full = scaler.transform(X) # Scale the full dataset for PCA visualization
X_pca = pca.fit_transform(X_scaled_full)

# Create a DataFrame for PCA results for easier plotting
pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['status'] = y.values # Add the original status labels

print(f"Explained variance ratio by principal components: {pca.explained_variance_ratio_}")

# Visualize PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='status',
                palette='viridis', data=pca_df, s=100, alpha=0.7)
plt.title('PCA of Parkinson\'s Disease Dataset (2 Components)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True)
plt.legend(title='Status', loc='best', labels=['Healthy (0)', 'Parkinson\'s (1)'])
plt.show()

print("\nPCA visualization complete. This plot helps understand the separability of the classes in a reduced dimension.")

# --- End of Notebook ---
