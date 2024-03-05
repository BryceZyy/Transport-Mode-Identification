from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.utils.np_utils import to_categorical

def create_bilstm_model(input_shape, num_classes):
    """
    Create and compile a BiLSTM model.
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(64, input_shape=input_shape)))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

def train_bilstm_model(model, X_train, y_train, epochs=50, batch_size=72):
    """
    Train the BiLSTM model.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model
