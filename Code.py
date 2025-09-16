# Parkinson's diesease Prediction
# A Machine Learning mini Project 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Class: Parkinson's Predictor
class ParkinsonsPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.model_2d = None
    
    def load_data(self, file_path):
        """Load data from CSV file"""
        self.data = pd.read_csv(file_path)
        print("Dataset Preview:")
        print(self.data.head())
        return self.data
    
    def preprocess_data(self):
        """Preprocess data by separating features and target"""
        X = self.data.drop(columns=['name', 'status'], axis=1)
        y = self.data['status']
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Feature Scaling
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_model(self):
        """Train the logistic regression model"""
        self.model = LogisticRegression()
        self.model.fit(self.X_train_scaled, self.y_train)
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model and print results"""
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred)
        
        # Output results
        print(f"\nModel Evaluation Results:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", class_report)
        
        return accuracy, conf_matrix, class_report, y_pred
    
    def visualize_results(self, y_pred=None):
        """Visualize results using PCA"""
        if y_pred is None:
            y_pred = self.model.predict(self.X_test_scaled)
            
        # Visualize using PCA (2D plot)
        self.pca = PCA(n_components=2)
        X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        X_test_pca = self.pca.transform(self.X_test_scaled)
        
        self.model_2d = LogisticRegression()
        self.model_2d.fit(X_train_pca, self.y_train)
        
        # Plot decision boundary
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=self.y_train, 
                   cmap='coolwarm', label="Training data")
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, 
                   cmap='coolwarm', marker='x', label="Predicted Test data")
        
        # Plot decision boundary
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                           np.arange(y_min, y_max, 0.1))
        Z = self.model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        plt.colorbar()
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title("Logistic Regression Decision Boundary")
        plt.legend()
        plt.show()
    
    def predict(self, features):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Scale the features
        scaled_features = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(scaled_features)
        probabilities = self.model.predict_proba(scaled_features)
        
        return prediction, probabilities
