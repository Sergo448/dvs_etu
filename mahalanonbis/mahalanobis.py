from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter
from operator import itemgetter
import heapq
import numpy as np

iris = datasets.load_iris()

X = iris.data
y = iris.target

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

"""
Классифицировать тестовую точку можно при помощи метода k-ближайших соседей.
Сначала нужно вычислить ковариационную матрицу для каждого класса,
затем определить k-ближайших соседей для тестовой точки 
(вычислив расстояния от неё до всех точек всех классов с учетом ковариационных матриц)
и отнести точку к классу с наибольшим количеством вхождений среди k ближайших соседей.

Далее представлен код программы для классификации методом k-ближайших соседей.
Причем:
— Обратная матрица: (COV+E)^{-1} (метрика Евклида — Махаланобиса);
— Используется квадрат расстояния (квадратный корень не имеет значения для конечного результата классификации, 
а без него программа работает немного быстрее).
"""


class MkNN:
    def __init__(self, k, classes, inv_cov_matrices):
        self.n = len(classes)
        self.k = k
        self.classes = classes
        self.inv_cov_matrices = inv_cov_matrices

    @staticmethod
    def mahalanobis_sqr(point_from, point_to, inverse_covariance_matrix):
        delta = point_from - point_to
        return max(np.float64(0), np.dot(np.dot(delta, inverse_covariance_matrix), delta))

    def _get_k_smallest(self, test_point):
        generator = (
            (MkNN.mahalanobis_sqr(test_point, point, inv_cov), i)
            for i, (class_, inv_cov) in enumerate(zip(self.classes, self.inv_cov_matrices))
            for point in class_
        )
        return heapq.nsmallest(self.k, generator, key=itemgetter(0))

    def predict(self, test_point):
        return heapq.nlargest(1, Counter((i for _, i in self._get_k_smallest(test_point))).items(),
                              key=lambda t: (t[1], -t[0]))[0][0]

    def predict_proba(self, test_point):
        most_common = Counter((i for _, i in self._get_k_smallest(test_point)))
        classes_proba = np.array([most_common.get(i, 0) for i in range(self.n)])
        return classes_proba / classes_proba.sum()

    def predict_all_max(self, test_point):
        p = self.predict_proba(test_point)
        return np.where(p == max(p))[0]


def main():
    # Тестовые точки, которые следует отнести к какому-либо из classes
    test_points = X_test
    # Классы с точками
    classes = [
        X_train, y_train
    ]
    # Число тренировочных точек
    n_train_points = sum(class_.shape[0] for class_ in classes)
    # Список матриц ковариаций для каждого класса
    cov_matrices = [np.cov(class_, rowvar=False, ddof=1) for class_ in classes]
    # Список обратных матриц ковариаций для каждого класса -- расстояние Евклида - Махаланобиса
    inv_cov_matrices = [np.linalg.inv(cov + np.identity(cov.shape[0])) for cov in cov_matrices]
    for test_point in test_points:
        print("Point:", test_point)
        # k от 1 до числа точек (не включая)
        for i in range(1, n_train_points):
            classifier = MkNN(i, classes, inv_cov_matrices)
            print(f"{i}nn:",
                  1 + classifier.predict(test_point),
                  classifier.predict_proba(test_point),
                  classifier.predict_all_max(test_point))


if __name__ == "__main__":
    main()
