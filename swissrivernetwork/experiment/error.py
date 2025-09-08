import numpy as np

# provide numpy error functions to reduce code duplications

class Error:
    
    '''
    computes the root mean squared error
    '''
    @staticmethod
    def rmse(y, y_hat):        
        return np.sqrt(np.mean((y-y_hat)**2))
    
    '''
    computes the average error
    '''
    @staticmethod
    def mae(y, y_hat):
        return np.mean(np.abs(y-y_hat))
    
    '''
    computes the nash-sutcliffe model efficiency coefficient
    '''
    @staticmethod
    def nse(y, y_hat):
        y_mean = np.mean(y)
        return 1.-(np.sum((y-y_hat)**2)/np.sum((y-y_mean)**2))
