import pandas as pd
import numpy as np
import random


class MySVM():
    # пока не дуал)
    def __init__(self, n_iter=10, learning_rate=0.001, weights=None, b=None, C=1, sgd_sample=None, 
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.b = b
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state  = random_state

    def __str__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X, y, verbose=False):
        
        random.seed(self.random_state)
        
        X_mat, y_vec = self._prepare_data(X, y)
        n_samples, n_features = X_mat.shape

        self.weights = np.ones(n_features)
        self.b = 1.0

        if verbose:
            self._log(0, self._calculate_loss(X_mat, y_vec, self.weights, self.b))
            
        
   

        for i in range(1, self.n_iter + 1):
            X_batch, y_batch = self._get_batch(X_mat, y_vec, self.sgd_sample)
            
            for xi, yi in zip(X_batch, y_batch):
                margin = yi * (xi @ self.weights + self.b)

                if margin >= 1:
                    grad_w = 2 * self.weights
                    grad_b = 0.0
                else:
                    grad_w = 2 * self.weights - self.C * yi * xi
                    grad_b = -yi * self.C

                self.weights = self.weights - self.learning_rate * grad_w
                self.b = self.b - self.learning_rate * grad_b

            if verbose and i % verbose == 0:
                self._log(i, self._calculate_loss(X_batch, y_batch, self.weights, self.b))

    def predict(self, X):
        X_mat = X.to_numpy(dtype=float)
        preds = np.sign(X_mat @ self.weights + self.b)
        return np.where(preds <= 0, 0, 1)

    def get_coef(self):
        return (self.weights, self.b)

    def _prepare_data(self, X, y):
        X_mat = X.to_numpy(dtype=float)
        y_vec = np.where(np.asarray(y).ravel() <= 0, -1, 1)
        return X_mat, y_vec

    def _log(self, i, loss):
        if i == 0:
            print(f"start | loss: {loss:.2f}")
        else:
            print(f"{i} | loss: {loss:.2f}")

    def _calculate_loss(self, X, y, weights, b):
        margins = y * (X @ weights + b)
        hinge = np.maximum(0, 1 - margins)
        return weights @ weights + hinge.mean() * self.C
    
    
    def _get_batch(self, X, y, sgd_sample):
        # Функция для разделения данных для стахостического градиентного спуска
        if self.sgd_sample is None:
            return X, y
        
        n_count = X.shape[0]

        if type(sgd_sample) == int:
            batch_size = sgd_sample
        elif type(sgd_sample) == float:
            batch_size = round(sgd_sample * n_count)
        else:
            batch_size = n_count
        
        sample_rows_idx = random.sample(range(n_count), batch_size)
        X_batch = X[sample_rows_idx]
        y_batch = y[sample_rows_idx]
        
        return X_batch, y_batch

