from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam


def create_bilstm_model(input_shape, num_classes):
    """
    Create and compile a BiLSTM model.

    Parameters:
    - input_shape: Tuple indicating the shape of the input data.
    - num_classes: Integer, number of output classes.

    Returns:
    - Compiled BiLSTM model.
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(32, input_shape=input_shape)))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_bilstm_model(model, X_train, y_train, epochs=50, batch_size=72):
    """
    Train the BiLSTM model.

    Parameters:
    - model: The BiLSTM model to train.
    - X_train: Training data features.
    - y_train: Training data labels.
    - epochs: Number of epochs to train for.
    - batch_size: Size of the batches of data.

    Returns:
    - Trained model.
    """
    # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2)
    return model, history
