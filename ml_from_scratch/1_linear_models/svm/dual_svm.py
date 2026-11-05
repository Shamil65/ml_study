import numpy as np


class MySVM():
    # SVM в двойственной постановке, обучение методом SMO.
    # C — штраф за нарушения зазора. Большое C (или np.inf) — почти hard margin.
    def __init__(self, C=1.0, tol=1e-3, max_passes=5, max_iter=1000):
        self.C = C
        self.tol = tol                    # допуск при проверке KKT
        self.max_passes = max_passes      # сколько проходов подряд без изменений до остановки
        self.max_iter = max_iter          # защита: максимум проходов всего
        self.a = None
        self.weights = None
        self.b = 0.0

    def __str__(self):
        return f"MySVM class: C={self.C}, tol={self.tol}, max_passes={self.max_passes}"

    def fit(self, X, y):
        X, y = self._prepare_data(X, y)
        n = X.shape[0]

        self.K = X @ X.T          # K[i][j] = x_i · x_j
        self.y = y
        self.a = np.zeros(n)
        self.b = 0.0

        passes = 0
        iters = 0

        while passes < self.max_passes and iters < self.max_iter:
            iters += 1
            changed = 0

            for i in range(n):
                E_i = self._error(i)

                # нарушено ли KKT для точки i
                too_close = y[i] * E_i < -self.tol and self.a[i] < self.C
                too_heavy = y[i] * E_i > self.tol and self.a[i] > 0
                if not (too_close or too_heavy):
                    continue

                # ошибки всех точек
                errors = [self._error(k) for k in range(n)]

                # вторые точки пары: сначала та, где ошибка сильнее всего отличается от E_i
                others = [k for k in range(n) if k != i]
                others.sort(key=lambda k: -abs(E_i - errors[k]))

                for j in others:
                    if self._update_pair(i, j, E_i, errors[j]):
                        changed += 1
                        break

            if changed == 0:
                passes += 1
            else:
                passes = 0

        # итоговые w и b
        self.weights = (self.a * y) @ X
        self.b = self._final_b(X, y)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.sign(X @ self.weights + self.b)
        return np.where(preds <= 0, 0, 1)

    def get_coef(self):
        return (self.weights, self.b)

    def _prepare_data(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.where(np.asarray(y).ravel() <= 0, -1, 1)
        return X, y

    def _f(self, k):
        # оценка точки k: сумма a_t * y_t * (x_t · x_k) + b
        return (self.a * self.y) @ self.K[:, k] + self.b

    def _error(self, k):
        return self._f(k) - self.y[k]

    def _update_pair(self, i, j, E_i, E_j):
        # Возвращает True, если a_i и a_j изменились.
        y, a, K = self.y, self.a, self.K
        a_i_old = a[i]
        a_j_old = a[j]

        # знаменатель шага
        eta = K[i, i] + K[j, j] - 2 * K[i, j]
        if eta < 1e-12:
            return False

        # новое a_j
        a_j_new = a_j_old + y[j] * (E_i - E_j) / eta

        # границы для a_j (оба числа должны остаться в [0, C])
        C = self.C
        if y[i] == y[j]:
            low = max(0.0, a_i_old + a_j_old - C)
            high = min(C, a_i_old + a_j_old)
        else:
            low = max(0.0, a_j_old - a_i_old)
            high = min(C, C + a_j_old - a_i_old)
        a_j_new = min(max(a_j_new, low), high)

        if abs(a_j_new - a_j_old) < 1e-5:
            return False

        # новое a_i
        a_i_new = a_i_old + y[i] * y[j] * (a_j_old - a_j_new)

        # новое b
        d_i = a_i_new - a_i_old
        d_j = a_j_new - a_j_old
        b1 = self.b - E_i - y[i] * d_i * K[i, i] - y[j] * d_j * K[i, j]
        b2 = self.b - E_j - y[i] * d_i * K[i, j] - y[j] * d_j * K[j, j]

        if 0 < a_i_new < C:
            self.b = b1
        elif 0 < a_j_new < C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        a[i] = a_i_new
        a[j] = a_j_new
        return True

    def _final_b(self, X, y):
        # b из точек, которые стоят ровно на границе (0 < a < C): y_s - w·x_s, берём среднее
        on_border = (self.a > 1e-6) & (self.a < self.C - 1e-6)
        if not on_border.any():
            return self.b
        return np.mean(y[on_border] - X[on_border] @ self.weights)
