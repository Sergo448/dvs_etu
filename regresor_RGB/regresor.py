import cv2 as cv
import numpy as np
from PIL import Image


class Regressor_RGB:
    """
    Реализуем класс регрессора для изображений формата RGB
    для того, чтобы устранить хроматическую аберрацию на цветном
    изображении.
    """

    def __init__(self, image):
        self.image = cv.imread(image, cv.IMREAD_COLOR)

    def load_image(self):
        img_array = np.asarray(self.image, dtype="int32")
        return img_array

    def make_vector_and_split(self):
        """
        Векторизируем изображение
        Создаем 3 различных вектора из исходного
        трансформируем изображение (объединяем)

        :return: transformed_image_by_merge, transformed_image_by_weights

        """
        img_arr = self.load_image()
        R_image = img_arr[2].copy()
        G_image = img_arr[1].copy()
        B_image = img_arr[0].copy()

        # merge the transformed channels back to an image
        transformed_image_by_merge = cv.merge((B_image, G_image, R_image))

        alpha = 0.25
        beta = 0.75
        transformed_image_by_weights = cv.addWeighted(self.image, alpha, transformed_image_by_merge, beta, 0.0)

        return transformed_image_by_merge, transformed_image_by_weights
