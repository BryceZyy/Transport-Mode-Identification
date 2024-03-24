# Transport-Mode-Identification
This is the code for the paper "A Hybrid Intercity Multimodal Transport Mode Identification Method Based on Mobility Features and Sequential Relations Mined from Cellular Signaling Data".

## System Configuration

To ensure compatibility and reproducibility, the following system configuration was used for development:

- **Operating System:** Windows 10 CentOS 6.0
- **Python Version:** 3.8.18

## Library Versions

This project depends on specific versions of TensorFlow and scikit-learn:

- **TensorFlow Version:** 2.4.0
- **scikit-learn Version:** 1.1.0

## Model Configurations

To ensure fairness in the experimental evaluation, each model is configured with comparable parameters as outlined below:

| Model       | Configuration                                             |
|-------------|-----------------------------------------------------------|
| KNN         | 3 neighbors                                               |
| RF          | Maximum depth: 10, Estimators: 200, Max features: 10      |
| XGBoost     | Estimators: 200, Max depth: 10, Learning rate: 0.01       |
| LSTM        | Units: 32, Epochs: 50, Batch size: 72, Optimizer: Adam    |
| BiLSTM      | Units: 32, Epochs: 50, Batch size: 72, Optimizer: Adam    |
| CNN-BiLSTM  | Adds convolutional, pooling layer, batch normalization to BiLSTM architecture |
| Hybrid      | Uses RFE predictions as features for BiLSTM               |

## Performance Overview

The following table summarizes the training and inference times (in seconds) for each model:

| Model       | Training Time (s) | Inference Time (s) |
|-------------|-------------------|--------------------|
| KNN         | 0.04              | 0.04               |
| XGBoost     | 8.21              | 0.08               |
| LSTM        | 45.33             | 2.1                |
| RF          | 67.46             | 0.56               |
| Bi-LSTM     | 68.34             | 1.54               |
| CNN-BiLSTM  | 80.85             | 1.05               |
| Hybrid      | 143.60            | 2.01               |


The table below shows the average performance of each model across different evaluation metrics. This summary provides a quick overview of how each model performs in various scenarios, including tolling freeways (FWY), toll-free state highways (HWY), high-speed railways (HSR), normal-speed railways (NSR), and static state (SS), along with the overall average.

| Model       | Overall Avg | FWY Avg | HWY Avg | HSR Avg | NSR Avg | SS Avg |
|-------------|-------------|---------|---------|---------|---------|--------|
| KNN         | 0.729       | 0.689   | 0.623   | 0.798   | 0.366   | 0.936  |
| XGBoost     | 0.891       | 0.916   | 0.763   | 0.890   | 0.832   | 0.960  |
| LSTM        | 0.895       | 0.900   | 0.777   | 0.897   | 0.867   | 0.960  |
| RF          | 0.885       | 0.912   | 0.754   | 0.875   | 0.819   | 0.961  |
| BiLSTM      | 0.890       | 0.896   | 0.769   | 0.887   | 0.860   | 0.958  |
| CNN-BiLSTM  | 0.895       | 0.895   | 0.815   | 0.841   | 0.825   | 0.955  |
| Hybrid      | 0.920       | 0.937   | 0.801   | 0.930   | 0.926   | 0.956  |
