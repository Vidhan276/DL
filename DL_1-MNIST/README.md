
# Assignment 1: MNIST Data Preprocessing and Visualization

## Overview

This assignment focuses on the fundamentals of data preprocessing and exploratory data analysis for deep learning. It uses the MNIST handwritten digit dataset to demonstrate dataset loading, pixel normalization, train-test splitting, and visualization.

The objective is to prepare raw image data for subsequent deep learning experiments while understanding its structure and class distribution.

## Objectives

- Set up the Python environment for deep learning experiments.
- Download and load the MNIST dataset.
- Examine image dimensions and class labels.
- Normalize image pixel values.
- Perform a stratified train-test split.
- Visualize sample handwritten digits.
- Analyze the distribution of digit classes.

## Dataset

**MNIST Handwritten Digit Dataset**

- Source: KaggleHub (`hojjatk/mnist-dataset`)
- Training images loaded: 60,000
- Image dimensions: 28 × 28 pixels
- Features per image: 784
- Classes: 10 digits (0–9)
- Image type: Grayscale

The notebook loads MNIST images and labels from IDX-format files using `mlxtend`.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- KaggleHub
- MLxtend
- Google Colab / Jupyter Notebook

## Implementation

### 1. Dataset Loading

The dataset is downloaded using KaggleHub and loaded through `loadlocal_mnist()`. Image data and corresponding labels are extracted for preprocessing.

### 2. Data Normalization

Pixel values are converted to floating-point numbers and normalized from the original range of 0–255 to 0–1.

```python
X = X.astype("float32") / 255.0
```

### 3. Train-Test Split

The loaded training dataset is divided using Scikit-learn:

- Training: 80% (48,000 images)
- Testing: 20% (12,000 images)
- Stratification: Based on digit labels
- Random state: 42

### 4. Data Visualization

The notebook generates:

- A visualization of an individual handwritten digit.
- A grid of 10 sample digit images.
- A histogram showing the digit class distribution.

### 5. Data Verification

The shapes of the training and testing arrays, label dimensions, and normalized pixel ranges are displayed.

## How to Run

1. Open `Assignment-1.ipynb` in Google Colab or Jupyter Notebook.
2. Install the required packages:

```bash
pip install kagglehub mlxtend numpy matplotlib scikit-learn
```

3. Execute the notebook cells sequentially.
4. Inspect the dataset information and visualizations.

The MNIST dataset is downloaded automatically.

## Learning Outcomes

- Understanding the structure of image datasets.
- Loading and preprocessing handwritten digit images.
- Applying normalization to numerical data.
- Creating reproducible and stratified data splits.
- Visualizing images and analyzing class distributions.
- Preparing data for future neural network training.

## Conclusion

This assignment demonstrates the essential preprocessing steps required before training an image classification model. The normalized MNIST data and visualizations establish a foundation for subsequent deep learning experiments.

**Note:** This practical covers dataset preparation and visualization; it does not train a neural network.

## Files

- `Assignment-1.ipynb` — Implementation notebook
- `DL_1-MNIST.pdf` — Assignment document
