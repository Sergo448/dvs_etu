import cv2 as cv
import numpy as np


class Regressor_RGB:
    """
    Реализуем класс регрессора для изображений формата RGB
    для того, чтобы устранить хроматическую аберрацию на цветном
    изображении.
    """

    def __init__(self, image):
        self.image = cv.imread(image, cv.IMREAD_COLOR)

    # Пока откажемся от перевода в вектор
    """
   def load_image(self):
        img_array = np.asarray(self.image, dtype="int32")
        # print(img_array)
        return img_array
    """

    def make_vector_and_split(self):
        """
        Векторизируем изображение
        Создаем 3 различных вектора из исходного
        трансформируем изображение (объединяем)

        :return: transformed_image_by_merge, transformed_image_by_weights

        """
        R_image, G_image, B_image = cv.split(self.image)
        """
        R_image = img_arr[2].copy()
        G_image = img_arr[1].copy()
        B_image = img_arr[0].copy()

        # print(R_image)
        # print(B_image)
        # print(G_image)
        """

        # merge the transformed channels back to an image
        _merge = cv.merge((R_image, G_image, B_image))

        transformed_image_by_merge = cv.addWeighted(self.image, 0.75, _merge, 0.05, 0)
        # В данный момент пока не отлажено
        """
        alpha = 0.25
        beta = 0.75
        transformed_image_by_weights = cv.addWeighted(self.image, alpha, transformed_image_by_merge, beta, 0.0)

        return transformed_image_by_merge, transformed_image_by_weights
        """
        return transformed_image_by_merge
        # return _merge