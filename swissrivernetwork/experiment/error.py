import numpy as np


# provide numpy error functions to reduce code duplications

class Error:

    @staticmethod
    def rmse(y, y_hat):
        """
        computes the root mean squared error
        """
        return np.sqrt(np.mean((y - y_hat) ** 2))


    @staticmethod
    def mae(y, y_hat):
        """
        computes the average error
        """
        return np.mean(np.abs(y - y_hat))


    @staticmethod
    def nse(y, y_hat):
        """
        computes the nash-sutcliffe model efficiency coefficient
        """
        y_mean = np.mean(y)
        return 1. - (np.sum((y - y_hat) ** 2) / np.sum((y - y_mean) ** 2))


def compute_errors(actual, prediction):
    from swissrivernetwork.experiment.error import Error
    rmse = Error.rmse(actual, prediction)
    mae = Error.mae(actual, prediction)
    nse = Error.nse(actual, prediction)
    return rmse, mae, nse
