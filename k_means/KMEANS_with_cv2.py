import os
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def kmeans(path, klusters):
    # read the image
    image = cv.imread(path)

    # Before we do anything, let's convert the image into RGB format:

    # convert to RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)

    # Let's try to print the shape of the resulting pixel values:

    # print(pixel_values.shape)

    # define stopping criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    _, labels, (centers) = cv.kmeans(pixel_values, klusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)


# show the image

"""
Saving image in a result root
/home/sergey/PycharmProjects/dvs_etu/k_means/results
"""

save_path = './/results'

print('Start program')
klusters = int(input('Please, enter k-value: '))

# User input for directory where files to search
expectedDir = './/images'
i = 1

for fileName_relative in glob.glob(expectedDir + "**//*.jpg", recursive=True):
    print("Full file name with directories: ", fileName_relative)
    # Now get the file name with os.path.basename
    fileName_absolute = os.path.basename(fileName_relative)
    print("Only file name: ", fileName_absolute)

    # Экземпляр класса, который решает нашу первую задачу

    result = kmeans(expectedDir, klusters=klusters)

    print(f'Получен результат работы модели К-средних: {fileName_absolute}')
    print(result[1:2])
    save_path = '.\\results'

    print(f'Запись результата работы модели {fileName_absolute}')

    completeName = os.path.join(save_path, f'results_{i}' + ".txt")
    with open(completeName, 'w') as f:
        f.write(str(result) + '\n')
        print(f'Запись результата парсига {fileName_absolute} в {completeName} прошла успешно')
    i = i + 1

