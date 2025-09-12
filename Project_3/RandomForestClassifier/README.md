# Random Forest Classifier

A professional implementation of Random Forest-based classifiers for multiclass classification tasks, following software engineering best practices and machine learning principles.

## 📚 Theoretical Foundation

### Random Forest Algorithm Theory

Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate classifier.

#### Key Components:

1. **Bootstrap Aggregating (Bagging)**: Each tree trained on random sample with replacement
2. **Random Feature Selection**: Each split considers only subset of features
3. **Majority Voting**: Final prediction based on tree consensus
4. **Out-of-Bag (OOB) Evaluation**: Unbiased error estimation using unused samples

#### Mathematical Formulation:

**For B trees and random feature subset m:**
```
ŷ = majority_vote{T₁(x), T₂(x), ..., Tᵦ(x)}
```

**Feature Importance:**
```
FI_j = (1/B) * Σᵦ₌₁ᴮ FI_j^(b)
```

**OOB Error:**
```
OOB_error = (1/n) * Σᵢ₌₁ⁿ I(ŷᵢ^(OOB) ≠ yᵢ)
```

Where:
- B = number of trees
- m = number of features per split (typically √p)
- FI_j^(b) = importance of feature j in tree b
- ŷᵢ^(OOB) = OOB prediction for sample i

## 🏗️ Project Structure

```
RandomForestClassifier/
├── config/                           # Configuration files
│   └── random_forest_experiment_1.yaml  # Experiment configuration
├── random_forest/                    # Core Random Forest package
│   ├── __init__.py                   # Package initialization and exports
│   ├── config_loader.py              # Configuration management with Pydantic
│   ├── model_builder.py              # Model architecture construction
│   └── random_forest_with_softmax.py # Complete Random Forest implementation
├── scripts/                          # Training and evaluation scripts
│   ├── train.py                      # Training pipeline
│   └── evaluate.py                   # Model evaluation
├── logs/                             # Training logs and metrics
├── models/                           # Saved models
│   └── saved_Models/                 # Trained model storage
└── README.md                         # This file
```

## 🚀 Quick Start

### Installation

Ensure you have the required dependencies:

```bash
pip install scikit-learn pandas numpy pydantic matplotlib seaborn joblib
```

### Basic Usage

```python
from random_forest import RandomForestConfig, RandomForestClassifier

# Load configuration
config = RandomForestConfig.from_yaml('config/random_forest_experiment_1.yaml')

# Create and train classifier
classifier = RandomForestClassifier(config)
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
```

### Training from Command Line

```bash
cd scripts
python train.py ../config/random_forest_experiment_1.yaml /path/to/your/dataset.csv
```

## 📊 Model Configuration

### Core Parameters

The Random Forest implementation supports comprehensive hyperparameter tuning:

```yaml
# Tree Ensemble Parameters
n_estimators: 100           # Number of trees in forest
criterion: "gini"           # Split quality measure
max_depth: null             # Maximum tree depth
min_samples_split: 2        # Minimum samples to split
min_samples_leaf: 1         # Minimum samples in leaf

# Randomness Control
max_features: "sqrt"        # Features per split
bootstrap: true             # Bootstrap sampling
random_state: 42           # Reproducibility seed

# Performance Optimization
n_jobs: -1                 # Parallel processing
oob_score: true            # Out-of-bag evaluation
class_weight: "balanced"   # Handle class imbalance
```

## 🔬 Theoretical Advantages

### Why Random Forest Works

1. **Variance Reduction**:
   - Individual trees have high variance
   - Averaging reduces variance without increasing bias
   - Law of Large Numbers ensures convergence

2. **Bias-Variance Tradeoff**:
   - Each tree can be grown deep (low bias)
   - Averaging many trees reduces variance
   - Optimal balance for many problems

3. **Robustness**:
   - Resistant to overfitting with sufficient trees
   - Handles missing values naturally
   - Less sensitive to outliers than single trees

4. **Feature Selection**:
   - Implicit feature selection through random sampling
   - Feature importance provides interpretability
   - Handles high-dimensional data well

### Strengths and Limitations

**Strengths:**
- Excellent out-of-the-box performance
- No need for feature scaling
- Handles mixed data types naturally
- Provides feature importance scores
- Robust to outliers and noise
- Built-in cross-validation (OOB)

**Limitations:**
- Can overfit with very noisy data
- Biased toward categorical variables with more levels
- Memory intensive for large datasets
- Less interpretable than single trees
- May not perform well on very high-dimensional sparse data

## 📈 Performance Analysis

### Hyperparameter Impact

| Parameter | Effect on Model | Tuning Guidelines |
|-----------|----------------|------------------|
| `n_estimators` | More trees → better performance, slower training | Start with 100, increase until OOB error plateaus |
| `max_depth` | Deeper trees → more complex patterns, overfitting risk | Use None for small datasets, limit for large ones |
| `min_samples_split` | Higher values → more conservative splits | Increase to reduce overfitting |
| `max_features` | Fewer features → more randomness, less correlation | sqrt(n) for classification, n/3 for regression |

### Feature Importance Analysis

Random Forest provides three types of feature importance:

1. **Gini Importance**: Default scikit-learn importance
2. **Permutation Importance**: More reliable but computationally expensive
3. **Drop-Column Importance**: Most reliable but very expensive

## 🛠️ Advanced Features

### Feature Importance Visualization

```python
# Get and plot feature importance
importance_dict = classifier.get_feature_importance(sort_by_importance=True, top_k=20)
classifier.plot_feature_importance(top_k=20, save_path='feature_importance.png')
```

### Tree Analysis

```python
# Get information about trees in the forest
tree_info = classifier.get_tree_info()
print(f"Average tree depth: {tree_info['avg_depth']}")
print(f"Average number of leaves: {tree_info['avg_leaves']}")
```

### Out-of-Bag Score Monitoring

```python
# Access OOB score for model validation
if classifier.config.oob_score:
    print(f"OOB Score: {classifier.oob_score_:.4f}")
```

### Hyperparameter Tuning

```python
from random_forest.model_builder import RandomForestBuilder

builder = RandomForestBuilder(config)
best_model, results = builder.hyperparameter_tuning(
    X_train, y_train,
    param_grid={
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
)
```

## 📋 Configuration Reference

### Essential Parameters

- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: None)
- `min_samples_split`: Minimum samples to split node (default: 2)
- `min_samples_leaf`: Minimum samples in leaf (default: 1)
- `max_features`: Features considered per split (default: "sqrt")

### Regularization Parameters

- `min_impurity_decrease`: Minimum impurity decrease for split
- `ccp_alpha`: Complexity parameter for pruning
- `min_weight_fraction_leaf`: Minimum weighted fraction in leaf

### Performance Parameters

- `n_jobs`: Parallel processing cores
- `random_state`: Random seed for reproducibility
- `warm_start`: Reuse previous solution
- `verbose`: Training verbosity level

## 🎯 Best Practices

### Model Training

1. **Start with defaults**: Random Forest works well out-of-the-box
2. **Monitor OOB score**: Use as validation metric
3. **Feature selection**: Use importance scores for dimensionality reduction
4. **Class balancing**: Use `class_weight='balanced'` for imbalanced data

### Hyperparameter Tuning

1. **n_estimators**: Increase until OOB error plateaus
2. **max_depth**: Limit for regularization on large datasets
3. **min_samples_split**: Increase to prevent overfitting
4. **max_features**: Use sqrt(n_features) as starting point

### Feature Engineering

1. **No scaling required**: Random Forest handles different scales
2. **Handle missing values**: Can work with NaN values
3. **Categorical encoding**: One-hot encoding or target encoding
4. **Feature selection**: Use built-in importance scores

## 📊 Example Results

Performance on various dataset types:

| Dataset Characteristics | Typical Performance | Best Use Cases |
|------------------------|-------------------|----------------|
| Tabular, Mixed Types | 85-95% accuracy | Finance, Healthcare |
| High-dimensional | 80-90% accuracy | Gene expression, Text |
| Image Features | 70-85% accuracy | Computer Vision |
| Time Series Features | 75-90% accuracy | Sensor data, IoT |

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `n_estimators` or use `max_samples`
2. **Slow Training**: Reduce `n_estimators` or increase `min_samples_split`
3. **Overfitting**: Increase `min_samples_split`, reduce `max_depth`
4. **Poor Performance**: Check feature engineering and class balance

### Performance Optimization

```python
# Memory-efficient configuration
config.max_samples = 0.8        # Use subset of samples per tree
config.n_jobs = -1              # Use all CPU cores
config.verbose = 1              # Monitor progress

# Regularization for overfitting
config.min_samples_split = 10   # Require more samples to split
config.min_samples_leaf = 5     # Require more samples in leaves
config.max_depth = 20           # Limit tree depth
```

## 🧪 Model Interpretation

### Feature Importance

```python
# Get feature importance rankings
importance = classifier.get_feature_importance(sort_by_importance=True)
for feature, score in list(importance.items())[:10]:
    print(f"{feature}: {score:.4f}")
```

### Partial Dependence Plots

```python
from sklearn.inspection import partial_dependence, plot_partial_dependence

# Plot partial dependence for top features
fig, ax = plt.subplots(figsize=(12, 8))
plot_partial_dependence(
    classifier.model, X_train, 
    features=[0, 1, 2],  # Top 3 features
    ax=ax
)
```

## 📚 References

1. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning.
3. Louppe, G. (2014). Understanding random forests: From theory to practice.
4. Probst, P., Wright, M. N., & Boulesteix, A. L. (2019). Hyperparameters and tuning strategies for random forest.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow PEP 8 coding standards
4. Add comprehensive tests
5. Submit a pull request

## 📧 Contact

For questions and support, please contact the Machine Learning Team.
