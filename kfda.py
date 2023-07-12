import util
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import pairwise_kernels


class KFDA:

    def __init__(self, output_dim, centre_data, regularization, reg_param, kernel, **kernel_args):

        self.output_dim = output_dim
        self.centre_data = centre_data
        self.regularization = regularization
        self.reg_param = reg_param
        self.kernel = kernel
        self.kernel_args = kernel_args

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        quotients, coefficients = util.optimization(X, y, self.output_dim, self.centre_data, self.regularization, self.reg_param, self.kernel, **self.kernel_args)   
        self.quotients = quotients
        self.coefficients = coefficients
        
    def transform(self, X):
        KX = pairwise_kernels(X, self.X_train, metric=self.kernel, **self.kernel_args)
        if self.centre_data:
            K = pairwise_kernels(self.X_train, metric=self.kernel, **self.kernel_args)
            KX = util.row_col_centre(KX, col_mat=K)
        low_dim_rep = KX @ self.coefficients.T
        return low_dim_rep

    def predict(self, X):

        X_train = self.transform(self.X_train)
        X_train = X_train[:, np.newaxis] if self.output_dim == 1 else X_train

        X_valid = self.transform(X)
        X_valid = X_valid[:, np.newaxis] if self.output_dim == 1 else X_valid

        classifier = NearestCentroid()
        classifier.fit(X_train, self.y_train)
        predictions = classifier.predict(X_valid)
        return predictions