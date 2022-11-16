# project name: dvs_etu
# version: 1.0
# file name: kmeans.py
# auther: Sergo448, 7193
# date: 15.11.2021
# Python version : 3.10

import os
from kmeans import K_means
import cv2 as cv


def writer(image):
    # Создание директории в которой будут храниться результаты
    if not os.path.isdir("results"):
        os.mkdir("results")

    save_path = 'results'

    print(f'Запись результата работы модели')

    path = os.path.join(save_path, 'result.jpg')
    cv.imwrite(path, image)


def main(classes_number, path_to_image):

    img = cv.imread(path_to_image)
    k_means = K_means(num_clussters=classes_number, img_path=path_to_image)
    img_clustered = k_means.clustering(img=img, num_clusters=classes_number)
    # cv.imshow("Clustered image", img_clustered)

    # SAVING
    # using funk writer
    path_to_results = './/results'
    writer(image=img_clustered)


k = int(input('Enter classes number: '))
# path_to_image должен быть r-строкой
path_to_image = r'.//images//lena.jpg'

if __name__ == "__main__":
    main(classes_number=k, path_to_image=path_to_image)

"""
Error:

RuntimeWarning: overflow encountered in ubyte_scalars
Начинает считать...

Finaly:
Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. Use QT_QPA_PLATFORM=wayland to run on Wayland anyway.

"""
