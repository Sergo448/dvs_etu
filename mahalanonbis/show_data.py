import heapq
import numpy as np
from numpy import dot
from collections import Counter
from operator import itemgetter, sub
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import colors, pyplot as plt
from mahalanobis import MkNN


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
        j += 1
        show_data_on_mesh(k=test_point, classes=classes, inv_cov_matrices=inv_cov_matrices)


if __name__ == "__main__":
    main()