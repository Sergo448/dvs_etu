import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
    Поскольку мы знаем, что мы сшиваем 2 изображения, давайте прочитаем их.
"""

path_R = r'//home//sergey//PycharmProjects//dvs_etu//regresor_RGB//3grad_images_plus_rectangles//R_image.jpeg'
path_G = r'//home//sergey//PycharmProjects//dvs_etu//regresor_RGB//3grad_images_plus_rectangles//G_image.jpeg'
path_B = r'//home//sergey//PycharmProjects//dvs_etu//regresor_RGB//3grad_images_plus_rectangles//B_image.jpeg'

img_ = cv.imread(path_R)
img = cv.imread(path_G)

img1 = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

"""
cv2.cvtColorпреобразует входное изображение RGB в форму в градациях серого.

Для сшивания изображений у нас есть следующие основные шаги:

    Вычислить sift-keypoints и дескрипторы для обоих изображений.
    Вычислить расстояния между каждым дескриптором в одном изображении и каждым дескриптором в другом изображении.
    Выберите верхние «m» соответствия для каждого дескриптора изображения.
    БегRANSACоценить гомографию
    Деформация для выравнивания швов
    Теперь сшить их вместе

Во-первых, мы должны выяснить особенности соответствия на обоих изображениях.
Эти наиболее подходящие функции служат основой для сшивания.
Мы извлекаем ключевые точки и просеиваем дескрипторы для обоих изображений следующим образом:
"""

# without error
shift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = shift.detectAndCompute(img1, None)
kp2, des2 = shift.detectAndCompute(img2, None)

"""
kp1 и kp2 являются ключевыми точками, des1 и des2 являются дескрипторами соответствующих изображений.

Теперь полученные дескрипторы на одном изображении также должны распознаваться на изображении. 
Мы делаем это следующим образом:
"""

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)


"""
BFMatcher() соответствует функциям, которые более похожи.
Когда мы устанавливаем параметр k = 2, мы просим knnMatcher выдавать 2 лучших совпадения для каждого дескриптора.

«Совпадения» - это список списков, где каждый подсписок состоит из «k» объектов. 

Часто в изображениях есть огромные шансы, что функции могут существовать во многих местах изображения.
Это может ввести в заблуждение нас, чтобы использовать тривиальные функции для нашего эксперимента. 
Таким образом, мы отфильтровываем все совпадения, чтобы получить лучшие. 

Таким образом, мы применяем коэффициент проверки с использованием 2 лучших совпадений, полученных выше.
Мы рассматриваем совпадение, если указанное ниже соотношение преимущественно больше указанного.
"""

# Apply ratio test
good = []
for m in matches:
    if m[0].distance < 0.5 * m[1].distance:
        good.append(m)
    matches = np.asarray(good)

"""
Пришло время выровнять изображения. 
Поскольку мы знаем, что матрица гомографии необходима для выполнения преобразования, 
а матрица гомографии требует как минимум 4 совпадений, мы делаем следующее.
"""

if len(matches[:, 0]) >= 4:
    src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
    H, masked = cv.findHomography(src, dst, cv.RANSAC, 5.0)
# print H
else:
    raise AssertionError('Can’t find enough keypoints.')

"""
И, наконец, последняя часть, сшивание изображений. 
Теперь, когда мы нашли гомографию трансформации, 
мы можем перейти к деформации и сшить их вместе:
"""

dst = cv.warpPerspective(img_, H, (img.shape[1] + img_.shape[1], img.shape[0]))

plt.subplot(122)
plt.imshow(dst)
plt.title('Warped Image')
plt.show()

plt.figure()
dst[0: img.shape[0], 0: img.shape[1]] = img
cv.imwrite('RG_image.jpg', dst)
plt.imshow(dst)
plt.show()