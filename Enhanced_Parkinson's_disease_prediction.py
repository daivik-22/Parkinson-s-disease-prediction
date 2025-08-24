import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve, precision_recall_curve, f1_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import urllib.request
import warnings
warnings.filterwarnings('ignore')

class EnhancedParkinsonsPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.pca = None
        self.feature_selector = None
        self.best_model = None
        self.feature_names = None
        
    def download_uci_data(self):
        """Download Parkinson's dataset from UCI ML Repository"""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        try:
            urllib.request.urlretrieve(url, "parkinsons.data")
            print("Dataset downloaded successfully!")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure you have internet connection or download manually from:")
            print("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")
            return False
    
    def load_data(self, file_path=None):
        """Load data from CSV file or download from UCI"""
        if file_path is None:
            # Try to download from UCI
            if self.download_uci_data():
                file_path = "parkinsons.data"
            else:
                raise FileNotFoundError("Could not download dataset. Please provide file path.")
        
        try:
            self.data = pd.read_csv(file_path)
            print("Dataset loaded successfully!")
            print(f"Dataset shape: {self.data.shape}")
            print("\nDataset Preview:")
            print(self.data.head())
            print("\nDataset Info:")
            print(self.data.info())
            print(f"\nClass distribution:")
            print(self.data['status'].value_counts())
            return self.data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        # Basic statistics
        print("\nDescriptive Statistics:")
        print(self.data.describe())
        
        # Check for missing values
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        
        # Feature correlation analysis
        plt.figure(figsize=(15, 12))
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        
        plt.subplot(2, 2, 1)
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Distribution of target variable
        plt.subplot(2, 2, 2)
        self.data['status'].value_counts().plot(kind='bar', color=['lightcoral', 'lightblue'])
        plt.title('Distribution of Parkinson\'s Status')
        plt.xlabel('Status (0: Healthy, 1: Parkinson\'s)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # Feature importance using correlation with target
        plt.subplot(2, 2, 3)
        feature_importance = abs(correlation_matrix['status']).sort_values(ascending=False)[1:]  # Exclude self-correlation
        top_features = feature_importance.head(10)
        top_features.plot(kind='barh')
        plt.title('Top 10 Features Correlated with Parkinson\'s Status')
        plt.xlabel('Absolute Correlation with Status')
        
        # Box plot for some key features
        plt.subplot(2, 2, 4)
        key_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)']
        for i, feature in enumerate(key_features):
            if feature in self.data.columns:
                plt.boxplot([self.data[self.data['status']==0][feature].dropna(),
                           self.data[self.data['status']==1][feature].dropna()],
                          positions=[i*2, i*2+0.5], widths=0.4)
        plt.xticks(range(0, len(key_features)*2, 2), key_features, rotation=45)
        plt.title('Feature Distribution by Status')
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self, use_feature_selection=True, n_features=15, scaler_type='standard'):
        """Enhanced preprocessing with feature selection options"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        # Separate features and target
        X = self.data.drop(columns=['name', 'status'], axis=1)
        y = self.data['status']
        self.feature_names = X.columns.tolist()
        
        print(f"Original feature count: {X.shape[1]}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Choose scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        
        # Fit and transform data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Feature selection
        if use_feature_selection:
            # Use SelectKBest with f_classif
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
            self.X_train_selected = self.feature_selector.fit_transform(self.X_train_scaled, self.y_train)
            self.X_test_selected = self.feature_selector.transform(self.X_test_scaled)
            
            # Get selected feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = [self.feature_names[i] for i in selected_indices]
            print(f"Selected {len(self.selected_features)} features: {self.selected_features}")
            
            return self.X_train_selected, self.X_test_selected, self.y_train, self.y_test
        else:
            return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_multiple_models(self, use_selected_features=True):
        """Train multiple models and compare performance"""
        if use_selected_features and hasattr(self, 'X_train_selected'):
            X_train = self.X_train_selected
            X_test = self.X_test_selected
        else:
            X_train = self.X_train_scaled
            X_test = self.X_test_scaled
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Train models and store results
        self.model_results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, self.y_train, cv=5, scoring='accuracy')
            
            self.model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'auc_score': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        # Find best model based on F1 score
        best_model_name = max(self.model_results.keys(), 
                            key=lambda k: self.model_results[k]['f1_score'])
        self.best_model = self.model_results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name}")
        return self.model_results
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """Perform hyperparameter tuning for selected model"""
        if hasattr(self, 'X_train_selected'):
            X_train = self.X_train_selected
        else:
            X_train = self.X_train_scaled
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'SVM':
            model = SVC(random_state=42, probability=True)
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        elif model_name == 'Logistic Regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        
        # Perform grid search
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def comprehensive_evaluation(self):
        """Comprehensive model evaluation with visualizations"""
        if not self.model_results:
            print("No models trained. Please train models first.")
            return
        
        # Print performance comparison
        print("\nModel Performance Comparison:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'AUC':<10} {'CV Mean±Std':<15}")
        print("-" * 80)
        
        for name, results in self.model_results.items():
            print(f"{name:<20} {results['accuracy']:<10.4f} {results['f1_score']:<10.4f} "
                  f"{results['auc_score']:<10.4f} {results['cv_mean']:.3f}±{results['cv_std']:.3f}")
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Performance comparison bar plot
        metrics = ['accuracy', 'f1_score', 'auc_score']
        model_names = list(self.model_results.keys())
        
        for i, metric in enumerate(metrics):
            ax = axes[0, i]
            values = [self.model_results[name][metric] for name in model_names]
            bars = ax.bar(model_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylim(0, 1)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
        
        # ROC curves
        ax = axes[1, 0]
        for name, results in self.model_results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['probabilities'])
            ax.plot(fpr, tpr, label=f"{name} (AUC = {results['auc_score']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True)
        
        # Precision-Recall curves
        ax = axes[1, 1]
        for name, results in self.model_results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, results['probabilities'])
            ax.plot(recall, precision, label=f"{name}")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True)
        
        # Confusion matrix for best model
        ax = axes[1, 2]
        cm = confusion_matrix(self.y_test, self.model_results[self.best_model_name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {self.best_model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # Detailed classification report for best model
        print(f"\nDetailed Classification Report - {self.best_model_name}:")
        print(classification_report(self.y_test, 
                                  self.model_results[self.best_model_name]['predictions']))
    
    def feature_importance_analysis(self):
        """Analyze feature importance for tree-based models"""
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            model = self.model_results[self.best_model_name]['model']
            
            if hasattr(self, 'selected_features'):
                feature_names = self.selected_features
            else:
                feature_names = self.feature_names
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f"Feature Importance - {self.best_model_name}")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.show()
            
            print(f"\nTop 10 Most Important Features ({self.best_model_name}):")
            for i in range(min(10, len(importances))):
                print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    def predict(self, features):
        """Make predictions on new data"""
        if self.best_model is None:
            raise ValueError("No model trained. Please train models first.")
        
        # Ensure features is 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale the features
        scaled_features = self.scaler.transform(features)
        
        # Apply feature selection if used
        if hasattr(self, 'feature_selector') and self.feature_selector is not None:
            scaled_features = self.feature_selector.transform(scaled_features)
        
        # Make prediction
        prediction = self.best_model.predict(scaled_features)
        probabilities = self.best_model.predict_proba(scaled_features)
        
        return prediction, probabilities
    
    def predict_single_patient(self, patient_data_dict):
        """Predict for a single patient using a dictionary of features"""
        # Convert dictionary to DataFrame
        patient_df = pd.DataFrame([patient_data_dict])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(patient_df.columns)
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Fill with median values from training data
            for feature in missing_features:
                if feature in self.data.columns:
                    patient_df[feature] = self.data[feature].median()
        
        # Reorder columns to match training data
        patient_df = patient_df[self.feature_names]
        
        prediction, probabilities = self.predict(patient_df.values)
        
        result = {
            'prediction': 'Parkinson\'s Disease' if prediction[0] == 1 else 'Healthy',
            'probability_healthy': probabilities[0][0],
            'probability_parkinsons': probabilities[0][1],
            'confidence': max(probabilities[0])
        }
        
        return result

# Example usage and demonstration
def demo_enhanced_parkinsons_predictor():
    """Demonstration of the enhanced Parkinson's predictor"""
    
    # Initialize predictor
    predictor = EnhancedParkinsonsPredictor()
    
    # Load data (will download from UCI if not provided)
    try:
        predictor.load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Perform EDA
    predictor.exploratory_data_analysis()
    
    # Preprocess data
    predictor.preprocess_data(use_feature_selection=True, n_features=15)
    
    # Train multiple models
    predictor.train_multiple_models()
    
    # Comprehensive evaluation
    predictor.comprehensive_evaluation()
    
    # Feature importance analysis
    predictor.feature_importance_analysis()
    
    # Hyperparameter tuning for best performing model
    predictor.hyperparameter_tuning(predictor.best_model_name)
    
    # Example prediction
    sample_patient = {
        'MDVP:Fo(Hz)': 119.992,
        'MDVP:Fhi(Hz)': 157.302,
        'MDVP:Flo(Hz)': 74.997,
        'MDVP:Jitter(%)': 0.00784,
        'MDVP:Jitter(Abs)': 0.00007,
        'MDVP:RAP': 0.0037,
        'MDVP:PPQ': 0.00554,
        'Jitter:DDP': 0.01109,
        'MDVP:Shimmer': 0.04374,
        'MDVP:Shimmer(dB)': 0.426,
        'Shimmer:APQ3': 0.02182,
        'Shimmer:APQ5': 0.0313,
        'MDVP:APQ': 0.02971,
        'Shimmer:DDA': 0.06545,
        'NHR': 0.02211,
        'HNR': 21.033,
        'RPDE': 0.414783,
        'DFA': 0.815285,
        'spread1': -4.813031,
        'spread2': 0.266482,
        'D2': 2.301442,
        'PPE': 0.284654
    }
    
    result = predictor.predict_single_patient(sample_patient)
    print(f"\nSample Patient Prediction:")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Probability of Parkinson's: {result['probability_parkinsons']:.3f}")

if __name__ == "__main__":
    demo_enhanced_parkinsons_predictor()
