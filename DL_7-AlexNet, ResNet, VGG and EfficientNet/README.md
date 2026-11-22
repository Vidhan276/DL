
# Assignment 7: Comparative Analysis of Pretrained CNN Architectures

## Overview

This assignment compares four convolutional neural network architectures using transfer learning for image classification on the CIFAR-10 dataset.

The architectures studied are:

- AlexNet
- VGG16
- ResNet50
- EfficientNetB0

The objective is to analyze differences in classification performance, model complexity, training efficiency, and inference speed.

## Objectives

- Understand popular CNN architectures.
- Implement transfer learning using pretrained models.
- Prepare CIFAR-10 images for ImageNet-based models.
- Adapt pretrained classifiers for new classes.
- Train and evaluate multiple architectures.
- Compare classification metrics and computational cost.
- Identify trade-offs between accuracy and efficiency.

## Dataset

**CIFAR-10 Image Classification Dataset**

The dataset contains ten image categories:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

### Dataset Configuration

| Partition | Images |
|---|---:|
| Training | 4,000 |
| Validation | 800 |
| Testing | 1,000 |

Balanced class-wise subsets are selected from CIFAR-10 for faster experimentation.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Data Preprocessing

The notebook performs:

- Image resizing to 224 × 224
- Conversion to PyTorch tensors
- Normalization using ImageNet mean and standard deviation
- Stratified class-wise sampling

Training-time augmentation includes:

- Random horizontal flipping
- Random rotation
- Brightness and contrast adjustment

## Architectures Compared

### 1. AlexNet

An early deep CNN architecture using convolutional layers followed by fully connected classification layers.

### 2. VGG16

A deeper CNN architecture using stacked convolutional layers to learn hierarchical image features.

### 3. ResNet50

A residual neural network that uses skip connections to support training of deeper architectures.

### 4. EfficientNetB0

A CNN architecture designed to balance network depth, width, and image resolution efficiently.

## Transfer Learning

The notebook loads pretrained ImageNet weights using Torchvision.

For each architecture:

1. Load the pretrained network.
2. Freeze the appropriate feature-extraction layers.
3. Replace the final classification layer to predict 10 CIFAR-10 classes.
4. Train the remaining trainable parameters.
5. Evaluate the adapted network on the test dataset.

The exact trainable layers depend on the architecture's classifier structure.

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 6 |
| Batch size | 32 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Learning-rate scheduler | StepLR |
| Scheduler step size | 3 |
| Scheduler gamma | 0.5 |

The notebook retains the best model state according to validation accuracy during training.

## Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Confusion matrix
- Classification report
- Training time
- Inference time per image
- Total and trainable parameter counts

## Comparative Analysis

The notebook generates:

### 1. Training Analysis

- Validation accuracy curves
- Validation loss curves
- Training versus validation accuracy

### 2. Performance Comparison

- Accuracy, precision, recall, and F1-score charts
- Normalized confusion matrices
- Per-class classification reports

### 3. Efficiency Analysis

- Model size versus classification accuracy
- Training time versus accuracy
- Inference speed comparison
- Parameter count comparison

### 4. Additional Visualizations

- Multi-metric radar chart
- Sample correct and incorrect predictions
- Automatically generated model leaderboard

## Generated Outputs

When the relevant cells execute, the notebook saves visualizations including:

```text
validation_curves.png
train_vs_val_all_models.png
metrics_comparison.png
confusion_matrices.png
efficiency_scatter.png
radar_comparison.png
sample_predictions.png
```

## How to Run

1. Open `Assignment-7.ipynb` in Google Colab or Jupyter Notebook.
2. Install the dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

3. Enable a GPU if available.
4. Run all notebook cells sequentially.
5. Review the final leaderboard and comparative visualizations.

The notebook automatically downloads CIFAR-10 and the required pretrained model weights.

## Learning Outcomes

- Understanding differences between CNN architectures.
- Applying transfer learning to image classification.
- Working with pretrained ImageNet models.
- Modifying classifier output layers.
- Evaluating models with multiple metrics.
- Comparing accuracy against computational efficiency.
- Understanding practical model selection trade-offs.

## Conclusion

This assignment demonstrates transfer learning using four established CNN architectures on CIFAR-10.

Rather than evaluating models solely by accuracy, the implementation also considers parameter count, training time, inference speed, and classification performance.

The comparison highlights the importance of choosing an architecture based on both predictive performance and deployment constraints.

## Files

- `Assignment-7.ipynb` — Comparative transfer learning implementation
- `README.md` — Assignment documentation

