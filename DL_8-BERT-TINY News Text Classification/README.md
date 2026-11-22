
# Assignment 8: Financial News Sentiment Classification Using BERT

## Overview

This assignment implements a financial text sentiment classification system called **FinSentBERT**.

The system fine-tunes a pretrained BERT model to classify financial tweets and headlines into three sentiment categories:

- Bearish
- Bullish
- Neutral

The assignment explores transformer-based natural language processing, transfer learning, class imbalance handling, evaluation, and model explainability.

## Objectives

- Understand BERT and transformer architectures.
- Perform financial text preprocessing.
- Implement a traditional NLP classification baseline.
- Fine-tune pretrained BERT for sentiment analysis.
- Address class imbalance using weighted loss.
- Compare classical machine learning and deep learning.
- Visualize learned text representations.
- Build an interactive sentiment prediction interface.

## Dataset

**Twitter Financial News Sentiment Dataset**

Source: Hugging Face

```text
zeroshot/twitter-financial-news-sentiment
```

The dataset contains financial text labeled as:

| Label | Sentiment |
|---|---|
| 0 | Bearish |
| 1 | Bullish |
| 2 | Neutral |

The original training split is further divided into training and validation partitions.

The dataset's official validation split is used as the held-out test set.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Gradio

## Data Preprocessing

The notebook includes:

- Removal of URLs
- Whitespace normalization
- Sentiment class distribution analysis
- Tweet length analysis
- Class-specific word frequency visualization
- BERT WordPiece tokenization
- Dynamic sequence padding
- Maximum token length selection based on dataset statistics

## Models Implemented

### 1. TF-IDF + Logistic Regression

A traditional NLP classification baseline is implemented using:

- TF-IDF vectorization
- Unigrams and bigrams
- Logistic Regression
- Balanced class weights

This establishes a benchmark for the transformer-based classifier.

### 2. Fine-Tuned BERT

The main model uses:

```text
bert-base-uncased
```

Architecture:

```text
Financial Text
      |
BERT WordPiece Tokenizer
      |
Pretrained BERT Encoder
      |
Classification Head
      |
Bearish / Bullish / Neutral
```

The implementation uses the 12-layer BERT-base architecture with a hidden dimension of 768.

**Note:** Although the repository folder contains "BERT-TINY" in its name, the actual notebook uses `bert-base-uncased`, not BERT-Tiny.

## Training Configuration

| Parameter | Value |
|---|---|
| Model | bert-base-uncased |
| Epochs | 4 |
| Training batch size | 32 |
| Evaluation batch size | 64 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Learning-rate schedule | Linear |
| Warm-up | 10% of training steps |
| Selection metric | Macro-F1 |

A custom Hugging Face Trainer applies class-weighted cross-entropy to address the dataset's class imbalance.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro-F1 score
- Precision
- Recall
- Classification report
- Normalized confusion matrix

Macro-F1 is particularly important because the sentiment classes are imbalanced.

## Experimental Results

The recorded notebook execution reports:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.8065 | 0.7466 |
| Fine-tuned BERT | 0.8639 | 0.8302 |

The fine-tuned BERT model achieves better accuracy and macro-F1 than the traditional baseline in this experiment.

*Results may vary across different training environments or reruns.*

## Model Explainability

The notebook includes:

### 1. Embedding Visualization

t-SNE is used to visualize BERT's learned text representations before and after fine-tuning.

### 2. Error Analysis

The most confident incorrect predictions are inspected to identify ambiguous or difficult financial text examples.

### 3. Attention Visualization

Attention weights from the final BERT layer are used to highlight tokens receiving attention from the classification token.

Attention visualizations provide a way to inspect model behavior, although attention alone does not establish causal explanations.

## Interactive Application

A Gradio interface allows users to enter financial text and obtain predicted probabilities for:

- Bearish
- Bullish
- Neutral

The notebook includes example headlines for testing the classifier.

## Model Saving

The fine-tuned model and tokenizer are saved to:

```text
finsentbert_best/
```

The folder is generated when the corresponding notebook cell executes.

## How to Run

1. Open `Assignment-8.ipynb` in Google Colab.
2. Enable GPU acceleration.
3. Install the dependencies:

```bash
pip install torch transformers datasets accelerate gradio numpy pandas matplotlib seaborn scikit-learn
```

4. Execute the notebook sequentially.
5. Train and evaluate the sentiment classifier.
6. Launch the Gradio interface to test predictions.

The pretrained BERT model and dataset are downloaded automatically.

## Learning Outcomes

- Understanding contextual language representations.
- Applying transformer-based NLP models.
- Fine-tuning pretrained BERT.
- Handling imbalanced classification datasets.
- Comparing traditional NLP with deep learning.
- Evaluating classifiers using macro-F1.
- Visualizing embeddings and attention patterns.
- Building interactive AI demonstrations.

## Conclusion

This assignment demonstrates how a pretrained transformer can be adapted for financial sentiment classification.

The experimental results show that fine-tuned BERT outperforms the TF-IDF and Logistic Regression baseline on both accuracy and macro-F1.

The implementation combines model training, quantitative evaluation, explainability techniques, and an interactive prediction interface.

## Files

- `Assignment-8.ipynb` — FinSentBERT implementation
- `README.md` — Assignment documentation

