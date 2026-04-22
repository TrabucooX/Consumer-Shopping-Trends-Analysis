# Consumer-Shopping-Trends-Analysis

## A fully functional ML model used to predict shopping preferences.
This project focuses on understanding the shopping tends between customer and their decisions. Additionally, a high-performance XGBoost classifier model is built to anticipate shopping trends between customers. The project does the following:

- Imports and encodes data for multilabel classification.
- Extracts a random set from the dataset to test the model.
- An XGBoost classifier model predicts the outputs.
- Results are shown with the visual help of a confusion matrix.

## Problems found during data analysis
The label used to make predictions is heavily imbalanced, thus stratify methods have been used during training to preserve these percentages. Other solutions have been to perform hyperparameter tuning on our XGBoost model with special attention at the following ones:
- "min_child_weight" : Used so that the model can be more conservative.
- "max_delta_step" : Used to control the steps we let each leaf to be inside our tree model, by adjusting it we can make the model more conservative, too.

By making the model more conservative, we force it to not learn the majority class by heart, thus helping with overfitting, so that it can also predict the least popular classes.

## Evaluation report and conclusion
By visualizing the results, we can see how the model distinguishes between labels:

![All Labels Confusion Matrix](./visuals/confusion_matrix.png)

As we can see the model performs fairly good on both the majority classes and the least popular ones. There is still some room for improvement in "Hybrid" class, which could be further improved by a more extensive hyperparameter tuning, training the model on a more balanced dataset or trying to collect more data with respect these last classes.