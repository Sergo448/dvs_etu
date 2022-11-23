from regresor import Regressor_RGB
from regresor_2 import Regressor_RGB as reg2
import os
import cv2 as cv


def writer(image_1, image_2=None):
    # Создание директории в которой будут храниться результаты
    if not os.path.isdir("results"):
        os.mkdir("results")

    save_path = 'results'

    print(f'Запись результата работы модели')

    path_1 = os.path.join(save_path, 'result_1.jpg')
    cv.imwrite(path_1, image_1)

    # Пока нет второго изображения
    # path_2 = os.path.join(save_path, 'result_2.jpg')
    # cv.imwrite(path_2, image_2)


def main(path_to_image):

    """
    regresor = Regressor_RGB(image=path_to_image)

    # by_weights not correct
    # transformed_image_by_merge, transformed_image_by_weights = regresor.make_vector_and_split()
    transformed_image_by_merge = regresor.make_vector_and_split()
    """
    regresor = reg2(image=path_to_image)
    transformed_image_by_merge = regresor.make_vector_and_split()

    # SAVING
    # using funk writer
    path_to_results = './/results'
    # writer(image_1=transformed_image_by_merge, image_2=transformed_image_by_weights)
    writer(image_1=transformed_image_by_merge, image_2=None)


path_to_image = r'.//images//image_with_aberation_2.jpg'

if __name__ == "__main__":
    main(path_to_image=path_to_image)

"""
Error:

Traceback (most recent call last):
  File "/home/sergey/PycharmProjects/dvs_etu/regresor_RGB/runner.py", line 36, in <module>
    main(path_to_image=path_to_image)
  File "/home/sergey/PycharmProjects/dvs_etu/regresor_RGB/runner.py", line 25, in main
    transformed_image_by_merge, transformed_image_by_weights = regresor.make_vector_and_split()
  File "/home/sergey/PycharmProjects/dvs_etu/regresor_RGB/regresor.py", line 39, in make_vector_and_split
    transformed_image_by_weights = cv.addWeighted(self.load_image(), alpha, transformed_image_by_merge, beta, 0.0)
cv2.error: OpenCV(4.6.0) /io/opencv/modules/core/src/arithm.cpp:647: error: (-209:Sizes of input arguments do not match)
 The operation is neither 'array op array' (where arrays have the same size and the same number of channels),
 nor 'array op scalar', nor 'scalar op array' in function 'arithm_op'

"""
