import numpy as np


# Returns the mean squared error or mean squared error derivative of two numpy arrays.
def mse(real: np.array, pred: np.array, d=False) -> float:
    num = real - pred
    if not d:
        return np.average(np.square(num))
    else:
        return np.average(2 * num)
