
# Assignment 3: Forward Propagation and Backpropagation Using TensorFlow/Keras

## Overview

This assignment demonstrates forward propagation and backpropagation in a feedforward neural network using TensorFlow and Keras.

The MNIST handwritten digit dataset is used to investigate how learning rates and training epochs influence optimization, convergence, and classification performance.

## Objectives

- Implement forward propagation in a neural network.
- Understand gradient computation using backpropagation.
- Train a neural network using TensorFlow/Keras.
- Experiment with different learning rates.
- Analyze the impact of training epochs.
- Compare fixed and decaying learning rates.
- Evaluate classification performance and errors.

## Dataset

**MNIST Handwritten Digit Dataset**

- Source: KaggleHub (`hojjatk/mnist-dataset`)
- Training samples: 60,000
- Testing samples: 10,000
- Image dimensions: 28 × 28
- Input features: 784 flattened pixels
- Output classes: 10 handwritten digits

Pixel values are normalized to the range 0–1.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- KaggleHub

## Neural Network Architecture

The model consists of:

| Layer | Configuration |
|---|---|
| Input | 784 features |
| Hidden layer 1 | Dense, 128 neurons, ReLU |
| Hidden layer 2 | Dense, 64 neurons, ReLU |
| Output | Dense, 10 neurons, Softmax |

**Optimizer:** Adam

**Loss Function:** Sparse Categorical Cross-Entropy

**Evaluation Metric:** Accuracy

## Implementation

### 1. Data Preprocessing

- Download MNIST using KaggleHub.
- Read image and label data from IDX files.
- Normalize pixel values.
- Flatten each 28 × 28 image into a 784-dimensional vector.
- Visualize sample digits and class distribution.

### 2. Forward Propagation

Input data passes through the dense layers. ReLU introduces nonlinearity, while Softmax produces class probabilities for the ten digits.

### 3. Backpropagation

An explicit training step uses TensorFlow's `GradientTape` to:

1. Perform a forward pass.
2. Calculate classification loss.
3. Compute gradients of the loss.
4. Update trainable parameters using Adam.

A custom training loop runs for 3 epochs and records loss, accuracy, and gradient norms.

### 4. Hyperparameter Experiments

The notebook evaluates combinations of:

**Learning rates:**
- 0.1
- 0.01
- 0.001
- 0.0001

**Training epochs:**
- 5
- 15
- 30

This creates 12 learning-rate/epoch configurations.

An additional experiment uses a learning rate of 1.0 to examine unstable optimization.

### 5. Learning Rate Scheduling

An exponential learning-rate decay schedule is implemented and compared against a fixed learning rate.

### 6. Evaluation and Visualization

The notebook includes:

- Training and validation performance analysis
- Learning-rate versus accuracy heatmap
- Learning-rate versus loss heatmap
- Validation accuracy and loss curves
- Fixed versus decaying learning-rate comparison
- Confusion matrix
- Classification report
- Misclassified digit examples
- Visualization of learned first-layer weight patterns

## How to Run

1. Open `Assignment-3.ipynb`.
2. Install the required dependencies:

```bash
pip install tensorflow kagglehub numpy pandas matplotlib seaborn scikit-learn
```

3. Run all cells sequentially.
4. Review the recorded training metrics and plots.

The hyperparameter experiments train multiple models and may be computationally intensive.

## Learning Outcomes

- Understanding forward and backward propagation.
- Computing gradients through automatic differentiation.
- Understanding how optimizers update neural network weights.
- Studying the effects of learning rate and epoch count.
- Identifying unstable training behavior.
- Evaluating neural network classification errors.

## Conclusion

This assignment demonstrates how backpropagation enables neural networks to learn through gradient-based optimization. The experiments illustrate the importance of choosing suitable learning rates and training durations for stable model convergence.

## Files

- `Assignment-3.ipynb` — Implementation notebook
- `DL_3.pdf` — Assignment document
