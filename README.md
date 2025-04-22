# Parkinson's-disease-prediction
A Machine Learning Project for Parkinson's disease prediction using Logistic Regression

This repository contains a machine learning project for predicting Parkinson's disease using logistic regression. The model analyzes various biomedical voice measurements to predict whether a person has Parkinson's disease.
About Parkinson's Disease
Parkinson's disease is a progressive neurological disorder that affects movement. Symptoms develop gradually, sometimes starting with a barely noticeable tremor in just one hand. The disorder also commonly causes stiffness or slowing of movement. Early detection through biomarkers can help in early intervention and management of the disease.
Features

Data preprocessing and feature scaling: Standardization of features to improve model performance
Logistic regression model: Implementation of logistic regression for binary classification
Model evaluation metrics: Comprehensive assessment using accuracy, confusion matrix, classification report, and ROC curves
Dimensionality reduction: PCA for visualization and feature importance analysis
Interactive Jupyter notebook: User-friendly interface for data exploration and visualization
Modular code structure: Reusable components for easy extension and modification

Dataset
This project uses the UCI ML Parkinson's Disease dataset. This dataset comprises a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, and each row corresponds to one of 195 voice recordings from these individuals.

Dataset Information
The main aim of the data is to discriminate healthy people from those with PD, according to "status" column which is set to 0 for healthy and 1 for PD.
The dataset includes features such as:

1.  name: Subject name and recording number
2.  MDVP(Hz): Average vocal fundamental frequency
3.  MDVP(Hz): Maximum vocal fundamental frequency
4.  MDVP(Hz): Minimum vocal fundamental frequency
And many more measurements...

Usage
Using the Jupyter Notebook
1.  Start Jupyter Notebook: bashjupyter notebook
2.  Navigate to notebooks/parkinsons_prediction.ipynb
3.  Upload your Parkinson's dataset CSV when prompted
4.  Run the cells to see the prediction results and visualizations

License:
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments:
UCI Machine Learning Repository for providing the Parkinson's dataset
scikit-learn team for the machine learning library
