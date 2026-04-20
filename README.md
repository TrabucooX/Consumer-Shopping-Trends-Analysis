# Consumer-Shopping-Trends-Analysis

## A fully functional ML model used to predict shopping preferences.
This project focuses on understanding the shopping tends between customer and their decisions. Additionally, a high-performance XGBoost classifier model is built to anticipate shopping trends between customers. The project does the following:

- Imports and encodes data for multilabel classification.
- Extracts a random set from the dataset to test the model.
- An XGBoost classifier model predicts the outputs.
- Results are shown with the visual help of a confusion matrix.

## Problems found during data analysis
The label used to make predictions is heavily imbalanced,
By visualizing the results, we can see how the model distinguishes between labels:

![All Labels Confusion Matrix](./visuals/confusion_matrix.png)