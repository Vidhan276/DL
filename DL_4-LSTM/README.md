

# Assignment 4: Time-Series Forecasting Using LSTM

## Overview

This assignment implements a Long Short-Term Memory (LSTM) neural network using PyTorch to forecast monthly sunspot activity.

The model learns temporal patterns from historical sunspot observations and predicts future values. The assignment demonstrates sequence preparation, LSTM architecture, model training, forecasting, and regression evaluation.

## Objectives

- Understand LSTM networks and sequential learning.
- Prepare time-series data for deep learning.
- Generate input sequences using sliding windows.
- Train an LSTM forecasting model.
- Evaluate prediction accuracy using regression metrics.
- Forecast multiple future time steps.
- Visualize actual and predicted values.

## Dataset

**Monthly Sunspots Dataset**

Source: Kaggle (`robervalt/sunspots`)

The dataset contains historical monthly mean total sunspot observations from approximately 1749 to 2017.

The primary columns used are:

- `Date`
- `Monthly Mean Total Sunspot Number`

The dataset is downloaded automatically using KaggleHub.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- KaggleHub

## Data Preprocessing

### 1. Chronological Splitting

The time series is divided without shuffling:

| Dataset | Percentage |
|---|---|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

### 2. Normalization

`MinMaxScaler` scales the data into the range 0–1.

The scaler is fitted only on the training partition to prevent information leakage.

### 3. Sliding-Window Creation

The model uses:

- Look-back window: 24 months
- Prediction horizon: 1 month

Each input sequence contains the previous 24 monthly observations, and the target is the following month's sunspot value.

## LSTM Architecture

The implementation uses a custom PyTorch `LSTMForecaster`.

| Component | Configuration |
|---|---|
| Input features | 1 |
| LSTM layers | 2 |
| Hidden units | 64 |
| Dropout | 0.2 |
| Intermediate dense layer | 32 neurons |
| Activation | ReLU |
| Output | 1 forecast value |

**Loss Function:** Mean Squared Error (MSE)

**Optimizer:** Adam

**Initial Learning Rate:** 0.001

## Model Training

The model is configured with:

- Maximum epochs: 100
- Batch size: 32
- Early stopping patience: 10 epochs
- Learning-rate scheduler: ReduceLROnPlateau
- Gradient clipping during training

The best model parameters are restored based on validation loss.

## Model Evaluation

Predictions are converted back to the original sunspot scale.

The notebook calculates:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

MAPE is calculated only for observations with more than 5 sunspots to reduce instability caused by values close to zero.

## Future Forecasting

The notebook implements recursive forecasting.

The trained model predicts one month ahead, then uses its previous predictions as input for subsequent forecasts.

**Forecast horizon:** 60 months (5 years)

## Visualizations

The notebook generates visualizations of historical sunspot activity, training and validation loss, test-set predictions, and future forecasts.

## Model Saving

The trained model, model configuration, and scaling parameters are saved to:

```text
lstm_forecaster_sunspots.pt
```

This file is generated when the corresponding notebook cell is executed.

## How to Run

1. Open `Assignment-4.ipynb` in Google Colab or Jupyter Notebook.
2. Install the dependencies:

```bash
pip install torch numpy pandas matplotlib scikit-learn kagglehub
```

3. Configure Kaggle API authentication if required by your environment.
4. Run the notebook sequentially.
5. Inspect the evaluation metrics and forecasting plots.

## Learning Outcomes

- Understanding LSTM-based sequence modeling.
- Preparing chronological training, validation, and testing data.
- Applying sliding-window forecasting.
- Preventing data leakage during normalization.
- Using early stopping and learning-rate scheduling.
- Evaluating time-series predictions.
- Implementing recursive multi-step forecasting.

## Conclusion

This assignment demonstrates the application of LSTM networks to time-series forecasting. It illustrates how historical sequential patterns can be used to estimate future values while emphasizing proper preprocessing, evaluation, and model optimization.

## Files

- `Assignment-4.ipynb` — Implementation notebook
- `README.md` — Assignment documentation
