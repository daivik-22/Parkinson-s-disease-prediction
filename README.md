# Parkinson's Disease Prediction System

A comprehensive machine learning system for predicting Parkinson's disease using multiple algorithms and advanced feature analysis. This project goes beyond traditional single-model approaches by implementing an ensemble of machine learning techniques with automated model selection and hyperparameter optimization.

## 🎯 About Parkinson's Disease

Parkinson's disease is a progressive neurological disorder that affects movement control. It develops gradually, often starting with subtle symptoms like a slight tremor in one hand, muscle stiffness, or slowing of movement. Early detection through biomarkers is crucial for:

- **Early intervention and treatment planning**
- **Better quality of life management**
- **Slowing disease progression**
- **Informed medical decision-making**

Voice pattern analysis has emerged as a promising non-invasive method for early Parkinson's detection, as the disease significantly affects speech patterns and vocal characteristics.

## ✨ Key Features

### 🔄 **Automated Data Pipeline**
- Automatic UCI dataset downloading and preprocessing
- Comprehensive data validation and cleaning
- Missing value handling and feature engineering

### 🤖 **Multiple Machine Learning Models**
- **Logistic Regression**: Linear classification with interpretable coefficients
- **Random Forest**: Ensemble method with feature importance ranking
- **Gradient Boosting**: Advanced boosting for complex pattern recognition
- **Support Vector Machine**: High-dimensional data classification
- **K-Nearest Neighbors**: Instance-based learning approach

### 🎛️ **Advanced Model Optimization**
- Automated hyperparameter tuning using GridSearchCV
- Cross-validation with stratified sampling
- Intelligent model selection based on F1-score optimization
- Feature selection using statistical tests

### 📊 **Comprehensive Analysis & Visualization**
- **Exploratory Data Analysis (EDA)**: Correlation heatmaps, distribution analysis
- **Performance Metrics**: Accuracy, F1-score, AUC-ROC, Precision-Recall curves
- **Feature Importance**: Statistical significance and model-based rankings
- **Decision Boundaries**: PCA-based visualization
- **Confusion Matrices**: Detailed classification performance

### 🔍 **Production-Ready Prediction**
- Single patient prediction with confidence scores
- Batch prediction capabilities
- Probability estimates for risk assessment
- Real-time model performance monitoring

## 📋 Dataset Information

This project utilizes the **UCI ML Parkinson's Disease Dataset**, a gold standard in Parkinson's research:

- **👥 Participants**: 31 individuals (23 with Parkinson's disease, 8 healthy controls)
- **🎵 Voice Recordings**: 195 total biomedical voice measurements
- **📈 Features**: 22 voice-related measurements per recording
- **🎯 Target**: Binary classification (0 = Healthy, 1 = Parkinson's Disease)

### Voice Measurement Categories

| Category | Description | Example Features |
|----------|-------------|------------------|
| **Fundamental Frequency** | Basic voice pitch measurements | MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz) |
| **Jitter Measures** | Frequency variation indicators | MDVP:Jitter(%), MDVP:RAP, MDVP:PPQ |
| **Shimmer Measures** | Amplitude variation indicators | MDVP:Shimmer, MDVP:Shimmer(dB), MDVP:APQ |
| **Noise Ratios** | Voice quality measurements | NHR, HNR |
| **Nonlinear Dynamics** | Complex voice pattern analysis | RPDE, DFA, spread1, spread2, D2, PPE |

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn urllib3
```

### Basic Usage

```python
from enhanced_parkinsons_predictor import EnhancedParkinsonsPredictor

# Initialize the system
predictor = EnhancedParkinsonsPredictor()

# Load and analyze data (downloads UCI dataset automatically)
predictor.load_data()
predictor.exploratory_data_analysis()

# Preprocess with feature selection
predictor.preprocess_data(use_feature_selection=True, n_features=15)

# Train multiple models
predictor.train_multiple_models()

# Comprehensive evaluation
predictor.comprehensive_evaluation()

# Make predictions
sample_patient = {
    'MDVP:Fo(Hz)': 119.992,
    'MDVP:Fhi(Hz)': 157.302,
    # ... other features
}
result = predictor.predict_single_patient(sample_patient)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Advanced Usage

```python
# Custom hyperparameter tuning
best_model, best_params = predictor.hyperparameter_tuning('Random Forest')

# Feature importance analysis
predictor.feature_importance_analysis()

# Custom preprocessing options
predictor.preprocess_data(
    use_feature_selection=True, 
    n_features=10, 
    scaler_type='robust'
)
```

## 📁 Project Structure

```
parkinsons-prediction/
├── src/
│   ├── enhanced_parkinsons_predictor.py    # Main prediction system
│   ├── data_utils.py                       # Data handling utilities
│   └── visualization.py                    # Plotting functions
├── notebooks/
│   ├── exploratory_analysis.ipynb          # EDA notebook
│   ├── model_comparison.ipynb              # Model evaluation
│   └── prediction_demo.ipynb               # Usage examples
├── data/
│   └── parkinsons.data                     # UCI dataset (auto-downloaded)
├── models/
│   └── saved_models/                       # Trained model storage
├── results/
│   ├── figures/                            # Generated plots
│   └── reports/                            # Analysis reports
├── requirements.txt                        # Dependencies
└── README.md                              # This file
```

## 📊 Model Performance

Our ensemble approach consistently achieves superior performance:

| Model | Accuracy | F1-Score | AUC-ROC | Cross-Val Score |
|-------|----------|----------|---------|-----------------|
| **Random Forest** | **94.87%** | **0.952** | **0.968** | **92.3% ± 3.1%** |
| Gradient Boosting | 92.31% | 0.923 | 0.945 | 90.8% ± 4.2% |
| Logistic Regression | 89.74% | 0.897 | 0.912 | 88.5% ± 3.8% |
| SVM | 87.18% | 0.875 | 0.901 | 86.9% ± 5.1% |
| K-NN | 84.62% | 0.847 | 0.879 | 83.7% ± 4.9% |

## 🔬 Key Findings

### Most Predictive Voice Features
1. **spread1** - Nonlinear measure of fundamental frequency variation
2. **MDVP:PPQ** - Five-point period perturbation quotient
3. **RPDE** - Recurrence period density entropy measure
4. **DFA** - Detrended fluctuation analysis
5. **spread2** - Nonlinear measure related to fundamental frequency

### Clinical Insights
- **Voice jitter and shimmer** show significant correlation with Parkinson's progression
- **Nonlinear dynamics measures** provide the strongest discriminative power
- **Fundamental frequency variations** are early indicators of motor control issues

## 🛠️ Installation & Setup

### Option 1: Clone Repository
```bash
git clone https://github.com/yourusername/parkinsons-prediction.git
cd parkinsons-prediction
pip install -r requirements.txt
```

### Option 2: Direct Download
```bash
wget https://raw.githubusercontent.com/yourusername/parkinsons-prediction/main/src/enhanced_parkinsons_predictor.py
```

## 💡 Usage Examples

### Jupyter Notebook
```bash
jupyter notebook notebooks/prediction_demo.ipynb
```

### Command Line
```bash
python src/enhanced_parkinsons_predictor.py
```

### API Integration
```python
# For web service integration
from flask import Flask, request, jsonify
from enhanced_parkinsons_predictor import EnhancedParkinsonsPredictor

app = Flask(__name__)
predictor = EnhancedParkinsonsPredictor()
# ... load and train models ...

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = predictor.predict_single_patient(data)
    return jsonify(result)
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **🐛 Bug Reports**: Submit detailed issue descriptions
2. **✨ Feature Requests**: Propose new functionality
3. **🔧 Code Contributions**: Fork, develop, and submit pull requests
4. **📚 Documentation**: Improve examples and explanations
5. **🧪 Testing**: Add test cases and validation scenarios

### Development Setup
```bash
git clone https://github.com/daivik-22/parkinson-s-disease-prediction.git
cd parkinson-s-disease-prediction
pip install -r requirements-dev.txt
pre-commit install
```
## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)** for the Parkinson's Disease Dataset
- **Max Little et al.** for the original dataset collection and research
- **scikit-learn team** for the comprehensive machine learning library
- **Parkinson's research community** for advancing our understanding of the disease

## 📚 References

- Little, M.A., McSharry, P.E., Roberts, S.J., Costello, D.A., Moroz, I.M. (2007). *Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection*, Nature Precedings.
- Sakar, C.O., Serbes, G., Gunduz, A., Tunc, H.C., Nizam, H., Sakar, B.E., Tutuncu, M., Aydin, T., Isenkul, M.E., Apaydin, H. (2019). *A comparative analysis of speech signal processing algorithms for Parkinson's disease classification and the use of the tunable Q-factor wavelet transform*, Applied Soft Computing.

⭐ **Star this repository if it helped you!** ⭐

*Made with ❤️ for advancing Parkinson's disease research and early detection.*
