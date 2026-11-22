
# Assignment 6: Convolutional Neural Network for Fruit and Vegetable Disease Classification

## Overview

This assignment implements a Convolutional Neural Network (CNN) using TensorFlow and Keras for image classification.

The model is developed using a fruit and vegetable disease dataset containing healthy, rotten, and diseased produce categories.

The assignment explores convolutional feature extraction, image preprocessing, data augmentation, classification, and performance evaluation.

## Objectives

- Understand CNN architecture and image feature extraction.
- Load and preprocess an image dataset.
- Analyze image dimensions and class distributions.
- Implement custom convolutional neural networks.
- Experiment with data augmentation.
- Train a multiclass image classifier.
- Evaluate predictions using classification metrics.
- Analyze incorrectly classified images.

## Dataset

**Fruit and Vegetable Disease (Healthy vs Rotten)**

Source: KaggleHub

Dataset identifier:

```text
muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
```

The dataset contains images organized into class-specific directories.

Class names are detected dynamically from the downloaded dataset.

## Technologies Used

- Python
- TensorFlow
- Keras
- KaggleHub
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Data Exploration

The notebook performs exploratory analysis through:

- Automatic class directory detection
- Image count per class
- Class imbalance analysis
- Healthy versus rotten/diseased grouping
- Sample image visualization
- Image dimension analysis

## Data Preprocessing

The implementation includes:

- Image resizing
- Pixel normalization to 0–1
- Training, validation, and testing partitions
- TensorFlow data pipelines
- Dataset caching and prefetching

### Data Augmentation

The notebook defines an augmentation pipeline using:

- Random horizontal flipping
- Random rotation
- Random zoom
- Random contrast

It also calculates class weights to examine class imbalance.

## CNN Architectures

Two CNN architectures are defined.

### 1. Original CNN

The initial model uses:

- Convolutional layers with 32, 64, 128, and 256 filters
- Batch normalization
- Max pooling
- Global average pooling
- Dense layer with 256 neurons
- Dropout of 0.4
- Softmax classification output

This architecture demonstrates a deeper feature-extraction design.

### 2. Lightweight CNN

A faster model is subsequently implemented with:

| Layer | Configuration |
|---|---|
| Input | 64 × 64 RGB |
| Conv2D | 16 filters, ReLU |
| MaxPooling2D | Pooling |
| Conv2D | 32 filters, ReLU |
| MaxPooling2D | Pooling |
| Conv2D | 64 filters, ReLU |
| MaxPooling2D | Pooling |
| GlobalAveragePooling2D | Feature aggregation |
| Dense | 64 neurons, ReLU |
| Dropout | 0.3 |
| Output | Softmax |

The notebook trains and evaluates this lightweight model.

## Training Configuration

- Framework: TensorFlow/Keras
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Sparse Categorical Cross-Entropy
- Metric: Accuracy
- Epochs: 5
- Batch size: 64
- Input dimensions: 64 × 64

For faster experimentation, the training and validation pipelines use limited batches and the lightweight model does not apply the previously defined data augmentation.

## Evaluation

The notebook evaluates the classifier using:

- Classification report
- Precision
- Recall
- F1-score
- Confusion matrix
- Misclassified image visualization

## Visualizations

The implementation includes:

1. Image count distribution by class
2. Healthy versus rotten/diseased distribution
3. Sample image grid
4. Image dimension distribution
5. Data augmentation examples
6. Training and validation accuracy/loss curves
7. Confusion matrix
8. Misclassified image examples

## How to Run

1. Open `Assignment-6.ipynb` in Google Colab or Jupyter Notebook.
2. Install the dependencies:

```bash
pip install tensorflow kagglehub numpy pandas matplotlib seaborn scikit-learn
```

3. Run the cells sequentially.
4. Allow KaggleHub to download the dataset.
5. Inspect the training curves and classification results.

A GPU is recommended for faster image processing and model training.

## Learning Outcomes

- Understanding convolution and pooling operations.
- Developing custom CNN architectures.
- Preparing multiclass image datasets.
- Understanding data augmentation and class imbalance.
- Comparing deeper and lightweight CNN designs.
- Evaluating image classification models.
- Performing error analysis using misclassified images.

## Conclusion

This assignment demonstrates how CNNs extract spatial features from images for multiclass classification.

It also explores the trade-off between model complexity and computational efficiency by defining a deeper CNN and training a lightweight alternative.

The experiment provides practical experience in developing image classification pipelines using TensorFlow and Keras.

## Files

- `Assignment-6.ipynb` — CNN implementation and evaluation
- `DL_6.pdf` — Assignment document
