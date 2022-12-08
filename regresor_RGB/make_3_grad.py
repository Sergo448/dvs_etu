import os
import cv2 as cv
import numpy as np


def main(path_to_image):
    image = cv.imread(path_to_image, cv.IMREAD_UNCHANGED)

    """
    RED CHANEL
    """
    red_channel = image[:, :, 2]
    # create empty image with same shape as that of src image
    red_img = np.zeros(image.shape)
    # assign the red channel of src to empty image
    red_img[:, :, 2] = red_channel

    """
    GREEN CHANEL
    """
    green_channel = image[:, :, 1]
    # create empty image with same shape as that of src image
    green_img = np.zeros(image.shape)
    # assign the red channel of src to empty image
    green_img[:, :, 1] = green_channel

    """
    BLUE CHANEL
    """
    blue_channel = image[:, :, 0]
    # create empty image with same shape as that of src image
    blue_img = np.zeros(image.shape)
    # assign the red channel of src to empty image
    blue_img[:, :, 0] = blue_channel

    """
    SAVING 3 Grad images
    """
    if not os.path.isdir("3grad_images"):
        os.mkdir("3grad_images")

    save_path = '3grad_images'

    print(f'Запись результата работы модели')

    path_R = os.path.join(save_path, 'R_image.png')
    path_G = os.path.join(save_path, 'G_image.png')
    path_B = os.path.join(save_path, 'B_image.png')

    # Сохраняем оттенки изображений
    cv.imwrite(path_R, red_img)
    cv.imwrite(path_G, green_img)
    cv.imwrite(path_B, blue_img)

    # Проведем склеивание панорамы красного изображения и зеленого.


path_to_image = r'.//images//image_with_aberation_2.jpg'

if __name__ == "__main__":
    main(path_to_image=path_to_image)
