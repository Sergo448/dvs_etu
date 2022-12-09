import os
import numpy as np
import cv2 as cv


def reader(path):
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    return img


"""
cv2.rectangle(img, pt1, pt2, color[, thickness[,lineType[,shift]]]) 

(0, 0)
---------> ^x^
|
|
|
|
V
<y>
# cv.rectangle(img=img, pt1=(shapes[0], 0), pt2=(shapes[0] + 150, shapes[1]), color=(0, 0, 0), thickness=-1)
img.shape    (rows, columns, channels)
"""


def draw_rectangle_right(img):
    nul = np.zeros((img.shape[0], 150, 3))
    return np.hstack((img, nul))


def draw_rectangle_left(img):
    nul = np.zeros((img.shape[0], 150, 3))
    return np.hstack((nul, img))


def main(pathR, pathG, pathB):
    red_img = reader(path=pathR)
    red_img_r = cv.copyMakeBorder(red_img, 0, 0, 0, 0, cv.BORDER_REPLICATE)
    red_img_l = cv.copyMakeBorder(red_img, 0, 0, 0, 0, cv.BORDER_REPLICATE)
    red_img_r = draw_rectangle_right(img=red_img_r)
    red_img_l = draw_rectangle_left(img=red_img_l)

    green_img = reader(path=pathG)
    green_img_r = cv.copyMakeBorder(green_img, 0, 0, 0, 0, cv.BORDER_REPLICATE)
    green_img_l = cv.copyMakeBorder(green_img, 0, 0, 0, 0, cv.BORDER_REPLICATE)
    green_img_r = draw_rectangle_right(img=green_img_r)
    green_img_l = draw_rectangle_left(img=green_img_l)

    blue_img = reader(path=pathB)
    blue_img_r = cv.copyMakeBorder(blue_img, 0, 0, 0, 0, cv.BORDER_REPLICATE)
    blue_img_l = cv.copyMakeBorder(blue_img, 0, 0, 0, 0, cv.BORDER_REPLICATE)
    blue_img_r = draw_rectangle_right(img=blue_img_r)
    blue_img_l = draw_rectangle_left(img=blue_img_l)

    """
        SAVING 3 Grad images
    """
    if not os.path.isdir("3grad_images_plus_rectangles"):
        os.mkdir("3grad_images_plus_rectangles")

    save_path = '3grad_images_plus_rectangles'

    print(f'Запись результата работы модели')

    path_R_r = os.path.join(save_path, 'R_image_r.jpeg')
    path_G_r = os.path.join(save_path, 'G_image_r.jpeg')
    path_B_r = os.path.join(save_path, 'B_image_r.jpeg')
    path_R_l = os.path.join(save_path, 'R_image_l.jpeg')
    path_G_l = os.path.join(save_path, 'G_image_l.jpeg')
    path_B_l = os.path.join(save_path, 'B_image_l.jpeg')

    # Сохраняем оттенки изображений
    cv.imwrite(path_R_r, red_img_r)
    cv.imwrite(path_G_r, green_img_r)
    cv.imwrite(path_B_r, blue_img_r)

    cv.imwrite(path_R_l, red_img_l)
    cv.imwrite(path_G_l, green_img_l)
    cv.imwrite(path_B_l, blue_img_l)


path_R = r'//home//sergey//PycharmProjects//dvs_etu//regresor_RGB//3grad_images//R_image.jpeg'
path_G = r'//home//sergey//PycharmProjects//dvs_etu//regresor_RGB//3grad_images//G_image.jpeg'
path_B = r'//home//sergey//PycharmProjects//dvs_etu//regresor_RGB//3grad_images//B_image.jpeg'

"""
path_R = r'..//3grad_images//R_image.jpeg'
path_G = r'..//3grad_images//G_image.jpeg'
path_B = r'..//3grad_images//B_image.jpeg'
"""

if __name__ == "__main__":
    main(pathR=path_R, pathG=path_G, pathB=path_B)
