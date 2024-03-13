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

    lenth = X.shape[1] - 1
    mat = X[0, 1:]
    fillmat = np.full([look_back - 1, lenth], 0)
    dataX = np.vstack((fillmat, mat))
    premdn = X[0, 0]
    i = 1
    count = 1
    while i < len(X):
        if (X[i][0] == premdn):
            if (count < look_back):
                temp = X[i - count:i + 1, 1:]
                filltemp = fill_zeros(temp, look_back, lenth)
                dataX = np.vstack((dataX, filltemp))
            else:
                dataX = np.vstack((dataX, X[i - look_back + 1:i + 1, 1:]))
            i += 1
            count += 1
        else:
            mat = X[i, 1:]
            fillmat = np.full([look_back - 1, lenth], 0)
            fillmat = np.vstack((fillmat, mat))
            dataX = np.vstack((dataX, fillmat))
            premdn = X[i, 0]
            count = 1
            i += 1
    return np.array(dataX)

def fill_zeros(mat, look_back, lenth):
    if mat.shape[0] < look_back:
        temp = np.full([look_back - mat.shape[0], lenth], 0)
        temp = np.vstack((temp, mat))
        return temp
    else:
        return mat
