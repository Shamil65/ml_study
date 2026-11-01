import pandas as pd
import numpy as np
import random


class MyLogReg():
    # Класс для реализации логистической регрессии
    def __init__(self, n_iter=10, 
                 learning_rate=0.1, 
                 weights=None, 
                 metric=None, reg=None, 
                 l1_coef=0.0, 
                 l2_coef=0.0, 
                 sgd_sample=None, 
                 random_state=42):
        
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.reg = reg
        self.sgd_sample = sgd_sample
        self.random_state  = random_state


    def __str__(self):
        # Функция для вывода информации о модели
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"


    def sigmoid(self, z):
        # Функция для вычисления сигмоида
        return 1.0 / (1.0 + np.exp(-z))
    
    
    def _calculate_loss(self, X_mat, y_vec, weights, l1, l2):
        m = len(y_vec)
        eps = 1e-15
        y_pred = self.sigmoid(X_mat @ weights)
        
        penalty_l1 = l1 * np.sum(np.abs(weights))
        penalty_l2 = l2 * np.sum(weights ** 2)
        
        LogLoss = (-1/m) * np.sum(y_vec * np.log(y_pred + eps) + (1 - y_vec) * np.log(1 - y_pred + eps))
        return LogLoss + penalty_l1 + penalty_l2
    

    def _prepare_data(self, X, y):
        # Функция для подготовки данных
        ones_np_array = np.ones(len(X))
        ones_df = pd.DataFrame(ones_np_array, columns=["bias"], index=X.index)
        full_X = pd.concat([ones_df, X], axis=1)
        full_X_mat = full_X.values
        y_vec = y.values
        return full_X_mat, y_vec
    

    def _log(self, i, loss, metric_value=None):
        if i == 0:
            prefix = "start"
        else:
            prefix = str(i)
        
        output = f"{prefix} | loss: {loss:.2f}"
        
        if metric_value is not None:
            output += f" | {self.metric}: {metric_value:.2f}"
        
        print(output)


    def _get_batch(self, X, y, sgd_sample):
        # Функция для разделения данных для стахостического градиентного спуска
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


    def fit(self, X, y, verbose=False):
        
        random.seed(self.random_state)
        
        X_mat, y_vec = self._prepare_data(X, y)
        m_instances, n_features = X_mat.shape
        self.weights = np.ones(n_features)

        l1 = self.l1_coef if self.reg in ['l1', 'elasticnet'] else 0.0
        l2 = self.l2_coef if self.reg in ['l2', 'elasticnet'] else 0.0

        is_lr_callable = callable(self.learning_rate)

        for i in range(0, self.n_iter + 1):
            y_pred = self.sigmoid(X_mat @ self.weights)

            # Логируем и считаем метрику ТОЛЬКО на verbose-шагах
            if verbose and i % verbose == 0:
                loss = self._calculate_loss(X_mat, y_vec, self.weights, l1, l2)
                metric_value = self._calculate_metric(y_vec, y_pred)
                self._log(i, loss, metric_value)

            if i == 0:
                continue

            X_batch, y_batch = self._get_batch(X_mat, y_vec, self.sgd_sample)

            y_pred_batch = self.sigmoid(X_batch @ self.weights)
            errors = y_pred_batch - y_batch

            gradient = (X_batch.T @ errors) / len(y_batch)

            if is_lr_callable:
                self.learning_rate_iter = self.learning_rate(i)
            else:
                self.learning_rate_iter = self.learning_rate

            reg_gradient = gradient + l1 * np.sign(self.weights) + l2 * 2 * self.weights
            self.weights -= self.learning_rate_iter * reg_gradient



        # Метрика обученной (финальной) модели — считаем ОДИН раз
        final_pred = self.sigmoid(X_mat @ self.weights)

        final_metric = self._calculate_metric(y_vec, final_pred)
        if final_metric is not None:
            if self.best_score is None or final_metric > self.best_score:
                self.best_score = final_metric


    def _calculate_roc_auc(self, y_true, y_score_rounded):
        P = np.sum(y_true == 1)
        N = np.sum(y_true == 0)
        if P == 0 or N == 0:
            return 0

        # Средние ранги (обработка одинаковых скоров как в парной формуле с 0.5)
        order = np.argsort(y_score_rounded, kind='mergesort')
        s = y_score_rounded[order]
        n = len(y_score_rounded)
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j < n and s[j] == s[i]:
                j += 1
            ranks[order[i:j]] = (i + j + 1) / 2.0  # средний ранг (1-based)
            i = j

        roc_auc = (np.sum(ranks[y_true == 1]) - P * (P + 1) / 2.0) / (P * N)
        return roc_auc

        
    def _calculate_metric(self, y_true, y_pred_proba):
        # Если метрика не задана — выходим
        if self.metric == None:
            return None
        

        # y_true = [0, 1, 0, 0, 1, 0, 1, 1, 0]  
        # y_pred = [0, 0, 0, 0, 1, 0, 1, 1, 0]
        # 
        # TP (True Positive) – истинно положительные примеры;
        # TN (True Negative) – истинно отрицательные примеры;
        # FP (False Positive) – ложноположительные примеры;
        # FN (False Negative) – ложноотрицательные примеры.
        # 
        # TP - 3 | FP - 0
        # FN - 1 | TN - 5


        # Получаем бинарные предсказания по порогу 0.5
        y_pred_bin = y_pred_proba > 0.5

        TP = np.sum((y_true == 1) & (y_pred_bin == 1))
        TN = np.sum((y_true == 0) & (y_pred_bin == 0))
        FP = np.sum((y_true == 0) & (y_pred_bin == 1))
        FN = np.sum((y_true == 1) & (y_pred_bin == 0))


        # print(f"TP: {TP}\nTN: {TN}\nFP: {FP}\nFN: {FN}")

        # Расчет метрик accuracy, precision, recall, f1
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    

        if self.metric == 'accuracy':
            return accuracy
        
        if self.metric == 'precision':
            return precision
        
        if self.metric == 'recall':
            return recall
        
        if self.metric == 'f1':
            return f1

        if self.metric == 'roc_auc':
            y_score_rounded = np.round(y_pred_proba, 10)
            roc_auc_value = self._calculate_roc_auc(y_true, y_score_rounded)
            return roc_auc_value
        

    def get_best_score(self):
        return self.best_score


    def get_coef(self):
        # Функция для вывода коэффициентов
        return self.weights[1:]
    
    def predict(self, X):
        # predict – переводит вероятности в бинарные классы по порогу > 0.5
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        X_math = X_with_bias.values

        y_pred = self.sigmoid(X_math @ self.weights)
        y_pred_bin = y_pred > 0.5
        return y_pred_bin


    def predict_proba(self, X):
        # predict_proba – возвращает вероятности (логиты прогнанные через функцию сигмоиды)
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        X_math = X_with_bias.values

        y_pred = self.sigmoid(X_math @ self.weights)

        return y_pred
    



import pandas as pd
import numpy as np

# Создаём простой датасет
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
    'feature2': [0, 1, 0, 1, 0, 1, 0, 1],
    'target':   [0, 0, 0, 0, 1, 1, 1, 1]
})

X = df[['feature1', 'feature2']]
y = df['target']

model = MyLogReg(n_iter=100, learning_rate=0.1, metric='accuracy')
model.fit(X, y, verbose=10)
print("Best score:", model.get_best_score())