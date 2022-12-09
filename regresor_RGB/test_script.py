import cv2
import numpy as np


def cvshow(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des


def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # des1 - изображение шаблона, des2 - соответствующее изображение
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # Инициализировать изображение визуализации, соединить изображения A и B слева и справа вместе
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # Совместное прохождение, рисование совпадающих пар
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # После успешного совпадения пары точек нарисуйте ее на визуализации
        if s == 1:
            # Нарисуйте совпадающие пары
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # Вернуть результаты визуализации
    return vis


# Панорамное сшивание
def siftimg_rightlignment(img_right, img_left):
    _, kp1, des1 = sift_kp(img_right)
    _, kp2, des2 = sift_kp(img_left)
    goodMatch = get_good_match(des1, des2)
    # Когда совпадающие пары элементов фильтра больше 4 пар: вычислить матрицу перспективного преобразования
    if len(goodMatch) > 4:
        # Получить координаты точки соответствующей пары
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
        # Функция этой функции состоит в том, чтобы сначала использовать
        # RANSAC для выбора четырех лучших наборов точек сопряжения,
        # а затем вычислить матрицу H. H - это матрица 3 * 3

        # Измените угол обзора справа от изображения, и в результате получится преобразованное изображение
        result = cv2.warpPerspective(img_right, H, (img_right.shape[1] + img_left.shape[1], img_right.shape[0]))
        cvshow('result_medium', result)
        # Передайте картинку слева в левый конец результирующего изображения
        result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
        return result


path_R = r'..//3grad_images//R_image.jpeg'
path_G = r'..//3grad_images/G_image.jpeg'
path_B = r'..//regresor_RGB//3grad_images//B_image.jpeg'
# Соответствие функций + панорамная строчка
path_1 = r'//home//sergey//PycharmProjects//dvs_etu//regresor_RGB//3grad_images_plus_rectangles//B_image_l.jpeg'
path_2 = r'//home//sergey//PycharmProjects//dvs_etu//regresor_RGB//3grad_images_plus_rectangles//G_image_r.jpeg'
# Прочтите сшитую картинку (обратите внимание на расположение левой и правой картинок)
# Преобразовать графику справа
img_right = cv2.imread(path_1)
img_left = cv2.imread(path_2)

img_right = cv2.resize(img_right, None, fx=0.5, fy=0.3)
# Убедитесь, что два изображения одинакового размера
img_left = cv2.resize(img_left, (img_right.shape[1], img_right.shape[0]))

kpimg_right, kp1, des1 = sift_kp(img_right)
kpimg_left, kp2, des2 = sift_kp(img_left)

# Отображение исходного изображения и изображения после обнаружения ключевой точки одновременно
# cvshow('img_left', np.hstack((img_left, kpimg_left)))
# cvshow('img_right', np.hstack((img_right, kpimg_right)))

goodMatch = get_good_match(des1, des2)

all_goodmatch_img = cv2.drawMatches(img_right, kp1, img_left, kp2, goodMatch, None, flags=2)

# goodmatch_img Установите, сколько goodMatch перед собой [: 10]
goodmatch_img = cv2.drawMatches(img_right, kp1, img_left, kp2, goodMatch[:10], None, flags=2)

# cvshow('Keypoint Matches1', all_goodmatch_img)
# cvshow('Keypoint Matches2', goodmatch_img)

# Склеиваем картинку в панораму
result = siftimg_rightlignment(img_right, img_left)
cvshow('result', result)