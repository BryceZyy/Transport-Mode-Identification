from data_preprocessing import load_data, prepare_data
from random_forest_model import train_random_forest, evaluate_model, plot_feature_importances
from bilstm_model import create_bilstm_model, train_bilstm_model
from model_evaluation import plot_confusion_matrix
from utils import create_dataset
import numpy as np
from keras.utils.np_utils import to_categorical

def main():
    # Load and prepare the data
    df1, df2, df3 = load_data()
    x1, y1 = prepare_data(df1)
    x2, y2 = prepare_data(df2)
    x3, y3 = prepare_data(df3)

    # Random Forest model training and evaluation
    print("Training Random Forest model...")
    rf_model = train_random_forest(x1, y1)
    rf_metrics = evaluate_model(rf_model, x3, y3)
    print(f"Random Forest evaluation metrics: {rf_metrics}")
    
    # Optional: Display feature importances
    feature_names = x1.columns.tolist()
    plot_feature_importances(rf_model, feature_names)

    # Prepare data for the BiLSTM model
    look_back = 3
    trainX, trainY = create_dataset(np.array(x2), look_back), to_categorical(y2)
    testX, testY = create_dataset(np.array(x3), look_back), to_categorical(y3)

    # Ensure the data is in the correct shape for the model
    trainX = trainX.reshape((trainX.shape[0], look_back, -1))
    testX = testX.reshape((testX.shape[0], look_back, -1))

    # BiLSTM model training and evaluation
    print("Training BiLSTM model...")
    bilstm_model = create_bilstm_model(input_shape=(look_back, trainX.shape[2]), num_classes=5)
    bilstm_model = train_bilstm_model(bilstm_model, trainX, trainY, epochs=50, batch_size=72)
    
    # Predict and evaluate the BiLSTM model
    predictions = bilstm_model.predict(testX)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(testY, axis=1)
    
    # Visualize the confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=['Static', 'High-speed', 'Provincial Roads', 'High-speed Rail', 'Regular Rail'])

if __name__ == "__main__":
    main()
