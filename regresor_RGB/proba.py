import cv2 as cv
import numpy as np


def foo(path):
    img = cv.imread(path)
    nul = np.zeros((img.shape[0], 150, 3))
    img = np.hstack((img, nul))
    return img


"""
    nul = np.zeros((img.shape[0], 150, 3))
    img = np.hstack((nul, img))
    # img = np.hstack((img, nul))
    img = img / 255
    
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

"""
path = r"//home//sergey//PycharmProjects//dvs_etu//regresor_RGB//3grad_images//B_image.jpeg"


def main(path):
    img = foo(path=path)
    print(f'Запись результата работы модели')
    cv.imwrite('proba.jpeg', img)


if __name__ == "__main__":
    main(path=path)

