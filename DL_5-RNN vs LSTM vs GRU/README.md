
# Assignment 5: Comparative Analysis of RNN, LSTM, and GRU

## Overview

This assignment presents a comparative study of three recurrent neural network architectures: Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU).

The models are implemented using PyTorch and evaluated on a controlled synthetic sequence classification task designed to test their ability to learn long-range dependencies.

The primary objective is to understand how gating mechanisms in LSTM and GRU help address the limitations of conventional RNNs.

## Objectives

- Implement RNN, LSTM, and GRU classifiers.
- Understand sequential data processing.
- Explore long-range dependency learning.
- Study the vanishing-gradient problem.
- Compare classification performance.
- Analyze model complexity and computational efficiency.
- Evaluate performance across different sequence lengths.

## Dataset

**Synthetic Early-Signal Sequence Classification Dataset**

The dataset is generated programmatically.

| Property | Value |
|---|---|
| Total samples | 4,000 |
| Minimum sequence length | 15 |
| Maximum sequence length | 45 |
| Vocabulary size | 59 |
| Classification type | Binary |
| Training split | 70% |
| Validation split | 15% |
| Testing split | 15% |

Each sequence contains three important signal tokens positioned near its beginning, surrounded by neutral tokens.

The target label is determined by whether positive or negative signal tokens form the majority.

The model must preserve information from the beginning of the sequence to classify it using the final hidden state.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Model Architectures

Three architectures are implemented:

### 1. Vanilla RNN

Uses a recurrent hidden state without gating mechanisms.

### 2. LSTM

Uses input, forget, and output gates with a cell state to manage information over long sequences.

### 3. GRU

Uses update and reset gates to control information flow with a simpler architecture than LSTM.

### Common Architecture

```text
Input Token IDs
      |
Embedding Layer
      |
RNN / LSTM / GRU
      |
Final Hidden State
      |
Dropout Layer
      |
Fully Connected Layer
      |
Binary Classification Output
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dimension | 32 |
| Hidden dimension | 96 |
| Recurrent layers | 1 |
| Dropout | 0.3 |
| Batch size | 64 |
| Epochs | 20 |
| Learning rate | 0.002 |
| Optimizer | Adam |
| Loss function | BCEWithLogitsLoss |

Variable-length sequences are padded and processed using `pack_padded_sequence`.

## Implementation Workflow

1. Generate synthetic sequences with controlled long-range dependencies.
2. Divide the dataset into training, validation, and testing subsets.
3. Create custom PyTorch Dataset and DataLoader objects.
4. Implement RNN, LSTM, and GRU classifiers.
5. Train all three architectures using common hyperparameters.
6. Evaluate classification performance.
7. Analyze prediction accuracy for different sequence lengths.
8. Compare training time, inference time, and parameter counts.

## Evaluation Metrics

The notebook calculates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Training time
- Inference time
- Number of trainable parameters

## Experimental Results

The saved notebook output reports:

| Model | Test Accuracy | F1-score | ROC-AUC | Parameters |
|---|---:|---:|---:|---:|
| RNN | 0.532 | 0.546 | 0.546 | 14,465 |
| LSTM | 0.997 | 0.997 | 0.999 | 51,905 |
| GRU | 0.998 | 0.998 | 1.000 | 39,425 |

*These values are from the recorded notebook execution and may vary across reruns.*

### Key Observations

- The vanilla RNN struggles to preserve information across longer sequences.
- LSTM and GRU perform substantially better on this controlled classification task.
- GRU achieves the highest recorded F1-score.
- The vanilla RNN uses the fewest parameters.
- GRU uses fewer parameters than LSTM while achieving comparable performance.

## Visualizations

The notebook includes:

- Training and validation loss curves
- Training and validation accuracy curves
- Classification metric comparison
- Confusion matrices
- ROC curves
- Accuracy versus sequence length
- Parameter and computational efficiency comparisons
- Normalized performance heatmap

## How to Run

1. Open `Assignment--5.ipynb`.
2. Install the dependencies:

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn
```

3. Execute all notebook cells sequentially.
4. Examine the generated comparison tables and graphs.

The dataset is generated automatically, so no external dataset download is required.

## Learning Outcomes

- Understanding recurrent neural network architectures.
- Identifying the limitations of conventional RNNs.
- Understanding LSTM and GRU gating mechanisms.
- Handling variable-length sequences.
- Implementing sequence classification in PyTorch.
- Evaluating long-range dependency learning.
- Comparing model performance and efficiency.

## Conclusion

This assignment demonstrates the advantages of gated recurrent architectures for sequence classification involving long-range dependencies.

The recorded experiment shows that LSTM and GRU substantially outperform a vanilla RNN, while GRU offers a favorable balance between parameter count and classification performance.

The experiment highlights the importance of architectural choices when designing models for sequential data.

## Files

- `Assignment--5.ipynb` — Comparative implementation and analysis
- `README.md` — Assignment documentation

