from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Classifier and evaluate feature importances.
    
    Parameters:
    - X_train: Training data features
    - y_train: Training data labels
    
    Returns:
    - clf_model: Trained Random Forest model
    """
    clf = RandomForestClassifier(max_depth=9, n_estimators=200, max_features=10)
    clf_model = clf.fit(X_train, y_train)
    return clf_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using accuracy, precision, recall, and F1 score.
    
    Parameters:
    - model: The trained model
    - X_test: Test data features
    - y_test: Test data labels
    
    Returns:
    - A dictionary containing evaluation metrics: accuracy, precision, recall, F1 score
    """
    y_pred = model.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1': f1_score(y_test, y_pred, average='weighted')
    }
    return metrics

def plot_feature_importances(model, feature_names):
    """
    Plot the feature importances of the model.
    
    Parameters:
    - model: The trained model
    - feature_names: List of feature names for labeling the plot
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="r", align="center")
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

# Example usage (assuming X_train, y_train are defined and feature_names is a list of feature names)
# model = train_random_forest(X_train, y_train)
# metrics = evaluate_model(model, X_test, y_test)
# print(metrics)
# plot_feature_importances(model, feature_names)
