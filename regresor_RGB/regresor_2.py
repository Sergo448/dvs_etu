import cv2 as cv
import numpy as np
import os


def writer(image_):
    # Создание директории в которой будут храниться результаты
    if not os.path.isdir("results"):
        os.mkdir("results")

    save_path = 'results'

    print(f'Запись результата работы модели')

    path_ = os.path.join(save_path, 'result_.jpg')
    cv.imwrite(path_, image_)


# "Панорама" для красного и зеленого оттенка
def panorams_R_G(img_, img):
    img1 = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)

    img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    shift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = shift.detectAndCompute(img1, None)
    kp2, des2 = shift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5 * m[1].distance:
            good.append(m)
        matches = np.asarray(good)

    if len(matches[:, 0]) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    # print H
    else:
        raise AssertionError('Can’t find enough keypoints.')

    dst = cv.warpPerspective(img_, H, (img.shape[1] + img_.shape[1], img.shape[0]))

    dst[0: img.shape[0], 0: img.shape[1]] = img
    cv.imwrite('result_image_for_R_G.jpg', dst)
    result_ = dst

    return result_


# "Панорама" для синего и зеленого оттенка
def panorams_B_G(img_, img):
    img1 = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)

    img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    shift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = shift.detectAndCompute(img1, None)
    kp2, des2 = shift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5 * m[1].distance:
            good.append(m)
        matches = np.asarray(good)

    if len(matches[:, 0]) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    # print H
    else:
        raise AssertionError('Can’t find enough keypoints.')

    dst = cv.warpPerspective(img_, H, (img.shape[1] + img_.shape[1], img.shape[0]))

    dst[0: img.shape[0], 0: img.shape[1]] = img
    cv.imwrite('result_image_for_B_G.jpg', dst)


# "Панорама" приводящая все к уровню зеленого
def panorams_RGB(img_, img):
    img1 = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)

    img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    shift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = shift.detectAndCompute(img1, None)
    kp2, des2 = shift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5 * m[1].distance:
            good.append(m)
        matches = np.asarray(good)

    if len(matches[:, 0]) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    # print H
    else:
        raise AssertionError('Can’t find enough keypoints.')

    dst = cv.warpPerspective(img_, H, (img.shape[1] + img_.shape[1], img.shape[0]))

    dst[0: img.shape[0], 0: img.shape[1]] = img
    cv.imwrite('result_image_for_RGB.jpg', dst)


def main(path_to_image):
    image = cv.imread(path_to_image, cv.IMREAD_COLOR)

    R_image, G_image, B_image = cv.split(image)

    # Сохраняем оттенки изображений
    cv.imwrite('R_image.jpg', R_image)
    cv.imwrite('G_image.jpg', G_image)
    cv.imwrite('B_image.jpg', B_image)

    # Проведем склеивание панорамы красного изображения и зеленого.
    panorams_R_G(img=cv.imread('G_image.jpg'), img_=cv.imread('R_image.jpg'))
    panorams_B_G(img=cv.imread('G_image.jpg'), img_=cv.imread('B_image.jpg'))

    panorams_RGB(img=cv.imread('result_image_for_R_G.jpg'), img_=cv.imread('result_image_for_B_G.jpg'))

    res = panorams_RGB(img=cv.imread('result_image_for_R_G.jpg'), img_=cv.imread('result_image_for_B_G.jpg'))

    writer(image_=res)


path_to_image = r'.//images//image_with_aberation_2.jpg'

if __name__ == "__main__":
    main(path_to_image=path_to_image)
