import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.show()
