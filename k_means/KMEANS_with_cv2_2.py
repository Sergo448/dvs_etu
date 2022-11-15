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

    return segmented_image


klusters = int(input('Enter the number of k-value: '))
res = kmeans(path=r'//home//sergey//PycharmProjects//dvs_etu//k_means//images//lena.png', klusters=klusters)

# showing input image and result
image = cv.imread(r'//home//sergey//PycharmProjects//dvs_etu//k_means//images//lena.png')
img = cv.cvtColor(image, cv.COLOR_BGR2RGB)

figure_size = 15
plt.figure(figsize=(figure_size, figure_size))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(res)
plt.title('Segmented Image when K = %i' % klusters)
plt.xticks([])
plt.yticks([])

plt.savefig('res_figure.png')
