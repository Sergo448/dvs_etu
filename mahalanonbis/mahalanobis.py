import heapq
import numpy as np
from numpy import dot
from collections import Counter
from operator import itemgetter, sub
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import colors, pyplot as plt


class MkNN:
    def __init__(self, k, classes, inv_cov_matrices):
        self.n = len(classes)
        self.k = k
        self.classes = classes
        self.inv_cov_matrices = inv_cov_matrices

    @staticmethod
    def mahalanobis_sqr(point_from, point_to, inverse_covariance_matrix):
        delta = point_from - point_to
        return max(np.float64(0), dot(dot(delta, inverse_covariance_matrix), delta))

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

    # Визуализация классификации сетки точек


def show_data_on_mesh(k, classes, inv_cov_matrices):
    # Генерация сетки
    min_ = np.min([np.min(class_, axis=0) for class_ in classes], axis=1) - 1
    max_ = np.max([np.max(class_, axis=0) for class_ in classes], axis=1) + 1
    min_c = min(min_[0], min_[1])
    max_c = max(max_[0], max_[1])
    h = 0.05
    test_mesh = np.meshgrid(np.arange(min_c, max_c, h), np.arange(min_c, max_c, h))
    test_points = np.c_[test_mesh[0].ravel(), test_mesh[1].ravel()]
    # Классификация точек сетки
    classifier = MkNN(k, classes, inv_cov_matrices)
    test_mesh_labels = [sub(*classifier.predict_proba(x)) for x in test_points]
    # Создание графика
    plt.figure(figsize=(6, 5), dpi=90)
    class_colormap = colors.ListedColormap(['#070648', '#480607'])
    plt.pcolormesh(test_mesh[0], test_mesh[1],
                   np.asarray(test_mesh_labels).reshape(test_mesh[0].shape),
                   cmap='coolwarm', shading='nearest')
    plt.colorbar()
    plt.scatter([point[0] for class_ in classes for point in class_],
                [point[1] for class_ in classes for point in class_],
                c=[-i for i, class_ in enumerate(classes) for _ in class_],
                cmap=class_colormap)
    plt.axis([min_c, max_c, min_c, max_c])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("k=" + str(k))
    plt.show()


def main():
    iris = load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # Тестовые точки, которые следует отнести к какому-либо из classes

    array_0 = []
    array_1 = []
    array_2 = []

    for i, elem in enumerate(y_train):
        if elem == 0:
            array_0.append(list(X_train[i]))
        elif elem == 1:
            array_1.append(list(X_train[i]))
        elif elem == 2:
            array_2.append(list(X_train[i]))

    test_points = np.array(X_test)

    # Классы с точками
    classes = [
        # 0 class
        np.array(array_0),
        # 1 class
        np.array(array_1),
        # 2 class
        np.array(array_2)
    ]

    # Число тренировочных точек
    n_train_points = sum(class_.shape[0] for class_ in classes)
    # Список матриц ковариаций для каждого класса
    cov_matrices = [np.cov(class_, rowvar=False, ddof=1) for class_ in classes]
    # Список обратных матриц ковариаций для каждого класса -- расстояние Евклида - Махаланобиса
    inv_cov_matrices = [np.linalg.inv(cov + np.identity(cov.shape[0])) for cov in cov_matrices]

    j = 1
    for test_point in test_points:
        print("Point:", test_point, f'number : {j}')
        # k от 1 до числа точек (не включая)
        j += 1
        for i in range(1, n_train_points):
            classifier = MkNN(i, classes, inv_cov_matrices)
            """
            print(f"{i}nn:",
                  1 + classifier.predict(test_point),
                  classifier.predict_proba(test_point),
                  classifier.predict_all_max(test_point))
            """
            print(f"{i}nn:",
                  1+ classifier.predict(test_point),
                  classifier.predict_proba(test_point),
                  classifier.predict_all_max(test_point))

"""

Вывод программы в формате:
"knn: [наименьший номер класса (с 1), к которому можно отнести точку]
      [вероятностные оценки для тестовых точек для всех классов]
      [индексы классов (с 0), к которым можно отнести точку]".
"""
if __name__ == "__main__":
    main()


"""
        Для проверки 
        
(array([6.1, 2.8, 4.7, 1.2]), 1)
(array([5.7, 3.8, 1.7, 0.3]), 0)
(array([7.7, 2.6, 6.9, 2.3]), 2)
(array([6. , 2.9, 4.5, 1.5]), 1)
(array([6.8, 2.8, 4.8, 1.4]), 1)
(array([5.4, 3.4, 1.5, 0.4]), 0)
(array([5.6, 2.9, 3.6, 1.3]), 1)
(array([6.9, 3.1, 5.1, 2.3]), 2)
(array([6.2, 2.2, 4.5, 1.5]), 1)
(array([5.8, 2.7, 3.9, 1.2]), 1)
(array([6.5, 3.2, 5.1, 2. ]), 2)
(array([4.8, 3. , 1.4, 0.1]), 0)
(array([5.5, 3.5, 1.3, 0.2]), 0)
(array([4.9, 3.1, 1.5, 0.1]), 0)
(array([5.1, 3.8, 1.5, 0.3]), 0)
"""
