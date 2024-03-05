import numpy as np

def create_dataset(X, look_back):
    """
    Generate a dataset where each sample is a sequence constructed from the input data.
    The sequence is formed based on a look_back parameter and the continuity of a specific identifier (premdn).

    Parameters:
    - X: numpy array, input data from which sequences are created. Assumes the first column is the identifier.
    - look_back: int, number of previous time steps to include in the output sequence.

    Returns:
    - dataX: numpy array containing sequences of data.
    """
    length = X.shape[1] - 1  # Exclude the identifier column from the feature set
    dataX = []
    i = 0
    while i < len(X):
        if i + look_back <= len(X):
            # Check if the sequence breaks within the look_back period due to a change in identifier
            if len(set(X[i:i+look_back, 0])) == 1:
                sequence = X[i:i+look_back, 1:]  # Exclude identifier from features
                dataX.append(sequence)
            else:
                # Handle sequences crossing identifier boundaries
                sequence = fill_zeros(X[i, 1:].reshape(1, -1), look_back, length)
                dataX.append(sequence)
        i += 1
    return np.array(dataX)

def fill_zeros(mat, look_back, length):
    """
    Pad the input matrix with zeros on top to ensure it has the desired number of rows (look_back).
    This is used for handling sequences shorter than the look_back period.

    Parameters:
    - mat: numpy array, the matrix to be padded.
    - look_back: int, desired number of rows in the output matrix.
    - length: int, number of features in the input matrix.

    Returns:
    - numpy array, the padded matrix with dimensions (look_back, length).
    """
    if mat.shape[0] < look_back:
        zero_pad = np.zeros((look_back - mat.shape[0], length))
        padded_mat = np.vstack((zero_pad, mat))
        return padded_mat
    else:
        return mat

