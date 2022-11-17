from regresor import Regressor_RGB
import os
import cv2 as cv


def writer(image_1, image_2):
    # Создание директории в которой будут храниться результаты
    if not os.path.isdir("results"):
        os.mkdir("results")

    save_path = 'results'

    print(f'Запись результата работы модели')

    path_1 = os.path.join(save_path, 'result_1.jpg')
    path_2 = os.path.join(save_path, 'result_2.jpg')
    cv.imwrite(path_1, image_1)
    cv.imwrite(path_2, image_2)


def main(path_to_image):

    regresor = Regressor_RGB(image=path_to_image)

    transformed_image_by_merge, transformed_image_by_weights = regresor.make_vector_and_split()

    # SAVING
    # using funk writer
    path_to_results = './/results'
    writer(image_1=transformed_image_by_merge, image_2=transformed_image_by_weights)


path_to_image = r'.//images//image_with_aberation.jpg'

if __name__ == "__main__":
    main(path_to_image=path_to_image)
