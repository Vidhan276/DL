
# Assignment 2: Multilayer Perceptron (MLP) for Wine Classification

## Overview

This assignment implements a Multilayer Perceptron (MLP) neural network for multiclass classification using the Wine dataset.

The model learns patterns from chemical features of wine samples and classifies them into three categories. Its performance is analyzed using classification metrics and visualizations.

## Objectives

- Understand the architecture and working of an MLP.
- Perform multiclass classification using neural networks.
- Apply feature standardization.
- Train and evaluate an MLP classifier.
- Analyze training loss and validation accuracy.
- Visualize decision boundaries and learned representations.

## Dataset

**Wine Recognition Dataset**

Source: `sklearn.datasets.load_wine()`

| Property | Details |
|---|---|
| Samples | 178 |
| Features | 13 |
| Classes | 3 |
| Training split | 75% |
| Testing split | 25% |

The dataset contains chemical measurements used to distinguish wine categories.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab / Jupyter Notebook

## Model Architecture

The implementation uses Scikit-learn's `MLPClassifier`.

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=25,
    random_state=42
)
```

### Architecture

- Input layer: 13 features
- Hidden layer 1: 64 neurons
- Hidden layer 2: 32 neurons
- Hidden layer 3: 16 neurons
- Output: 3 wine classes
- Activation: ReLU
- Optimizer: Adam
- Early stopping: Enabled

## Implementation Workflow

### 1. Data Preparation

The Wine dataset is loaded and divided into training and testing sets using stratified sampling.

### 2. Feature Standardization

`StandardScaler` standardizes the input features. The scaler is fitted on the training data and applied to the test data.

### 3. Model Training

The MLP classifier is trained with three hidden layers and early stopping to reduce unnecessary training and help control overfitting.

### 4. Performance Evaluation

The notebook calculates:

- Classification accuracy
- Confusion matrix
- Precision, recall, and F1-score
- Multiclass ROC curves and AUC scores
- Per-class accuracy

### 5. Visual Analysis

The implementation includes:

- Confusion matrix heatmap
- Training loss curve
- Validation accuracy curve
- PCA-based decision boundary
- Multiclass ROC curves
- t-SNE visualization of hidden-layer representations
- Input feature weight analysis
- Per-class accuracy chart

A separate MLP is trained using two PCA components specifically to visualize the decision boundary.

## How to Run

1. Open `Assignment-2.ipynb`.
2. Install the dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

3. Run the notebook sequentially.
4. Observe the evaluation metrics and visualization dashboard.

The Wine dataset is available directly through Scikit-learn.

## Learning Outcomes

- Understanding feedforward neural networks.
- Building MLP models for multiclass classification.
- Understanding the importance of feature standardization.
- Evaluating classifiers beyond accuracy.
- Visualizing nonlinear decision boundaries.
- Analyzing hidden representations using dimensionality reduction.

## Conclusion

This assignment demonstrates how a Multilayer Perceptron can learn nonlinear relationships in structured data. Multiple evaluation metrics and visualizations provide insight into its classification performance and learned feature representations.

## Files

- `Assignment-2.ipynb` — Implementation notebook
- `DL_2-MLP_Classifier.pdf` — Assignment document
