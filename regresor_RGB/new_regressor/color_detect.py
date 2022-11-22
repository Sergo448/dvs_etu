# -*- coding:utf-8 -*-
import cv2
import numpy as np


def _img_split_with_shadow(gray_img, threshold_value=180):
    """
         : param binary_img: изображение в градациях серого, прочитанное в
    :param img_show:
         : return: набор координат горизонтальных и вертикальных линий
    """
    h = gray_img.shape[0]
    w = gray_img.shape[1]

    # Сумма по строке
    sum_x = np.sum(gray_img, axis=1)
    # Сумма по столбцу
    sum_y = np.sum(gray_img, axis=0)

    h_line_index = np.argwhere(sum_x == 0)
    v_line_index = np.argwhere(sum_y == 0)

    h_line_index = np.reshape(h_line_index, (h_line_index.shape[0],))
    v_line_index = np.reshape(v_line_index, (v_line_index.shape[0],))

    h_line = []
    v_line = []

    for i in range(len(h_line_index) - 1):
        if h_line_index[i + 1] - h_line_index[i] > 2:
            h_line.append((0, h_line_index[i + 1], w - 1, h_line_index[i + 1]))
            h_line.append((0, h_line_index[i], w - 1, h_line_index[i]))

    for i in range(len(v_line_index) - 1):
        if v_line_index[i + 1] - v_line_index[i] > 2:
            v_line.append((v_line_index[i + 1], 0, v_line_index[i + 1], h - 1))
            v_line.append((v_line_index[i], 0, v_line_index[i], h - 1))

    return h_line, v_line


def _combine_rect(h_lines, v_lines):
    """
         : param h_lines: набор параллельных линий
         : param v_lines: коллекция вертикальных линий
         : return: возвращает прямоугольный набор h_lines и v_lines
    """
    rects = []

    x_axis = sorted(set([item[0] for item in v_lines]))
    y_axis = sorted(set([item[1] for item in h_lines]))

    point_list = []
    for y in y_axis:
        point = []
        for x in x_axis:
            point.append((y, x))
        point_list.append(point)

    for y_index in range(len(y_axis) - 1):
        for x_index in range(len(x_axis) - 1):
            area = abs((y_axis[y_index + 1] - y_axis[y_index]) * (x_axis[x_index + 1] - x_axis[x_index]))
            rects.append([(y_axis[y_index], x_axis[x_index],
                          y_axis[y_index + 1], x_axis[x_index + 1]), area])
    # Сортировать по площади в порядке убывания
    rects.sort(key = lambda  ele: ele[1], reverse=True)
    areas = [ele[1] for ele in rects]

    # Найти серийный номер с наибольшей смежной разницей
    max = -1
    index = 0
    for i in range(len(areas) - 1):
        dif = areas[i] - areas[i + 1]
        if max < dif:
            max = dif
            index = i + 1

    # rects Сортировать в порядке возрастания координат, чтобы порядок цветов соответствовал стандартной цветовой карте.
    rect_list = [ele[0] for ele in rects[0:index]]
    rect_list.sort(key = lambda  ele: ele[1])
    rect_list.sort(key = lambda ele: ele[0])

    # for i in range(len(rect_list) - 1):
    #     for j in range(0, len(rect_list) - 1 - i):
    #         if rect_list[j + 1][1] < rect_list[j][1] :
    #             rect_list[j], rect_list[j + 1] = rect_list[j + 1], rect_list[j]
    #
    # for i in range(len(rect_list) - 1):
    #     for j in range(0, len(rect_list) - 1 - i):
    #         if rect_list[j + 1][0] < rect_list[j][0]:
    #             rect_list[j], rect_list[j + 1] = rect_list[j + 1], rect_list[j]

    return rect_list


def img_split(img, img_show=False):
    """
         Разделите изображение цветовой карты, которое нужно протестировать, и получите список сегментированных прямоугольных изображений и входное изображение, требуемое формой уравнения регрессии: (4, 6, 3), формат пикселей: (b, g, r)
         : param img_file: изображение цветовой карты для тестирования
         : param img_show: показывать ли
         : return: прямоугольный список разделенных подизображений
    """
    # Заполните 10 пикселей каждый
    padding = 10
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
    binary = cv2.blur(binary, (5, 5))
    binary = cv2.bitwise_not(binary)
    binary = cv2.copyMakeBorder(binary, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # cv2.imshow('cece', binary)
    # cv2.waitKey()
    h = img.shape[0]
    w = img.shape[1]
    rate = h // w if h > w else w // h

    h_line_shadow, v_line_shadow = _img_split_with_shadow(binary)
    h_line = h_line_shadow
    v_line = v_line_shadow
    rects = _combine_rect(h_line, v_line)
    split_imgs = []

    # padding тоже, поэтому вам нужно вычесть значение отступа при позиционировании
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    color_img = np.zeros((4,6,3),dtype=np.uint8)
    for index, rect in enumerate(rects):
        rect_img = img[rect[0]:rect[2], rect[1]:rect[3]]
        color_img[index//6][index%6] = get_center_color(rect_img)
        # print(index, color_img[index//6][index%6])
        split_imgs.append(rect_img)

    if img_show:
        p = 0
        for rect in rects:
            cv2.rectangle(img, (rect[1], rect[0]), (rect[3], rect[2]), (0, 255, 0), 2)
            # Напишите метку на идентифицированном объекте
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(p), (rect[1] - 10, rect[0] + 10), font, 1, (0, 0, 255), 2)  # Плюс и минус 10 - настроить положение персонажа
            p += 1

        img = cv2.resize(img, (int(h * 0.7), int(h * 0.7 / rate)))
        cv2.imshow('cece', img)
        cv2.waitKey()

    return split_imgs, color_img

def get_center_color(img):
    """
         Рассчитать среднее значение (5, 5) пикселей в середине данного изображения
    :param img:
    :return:
    """
    w = img.shape[0]
    w = w//2
    h = img.shape[1]
    h = h//2
    data = img[h - 2:h + 2, w - 2:w + 2]
    b,g,r = cv2.split(data)
    return (int(np.mean(b)), int(np.mean(g)), int(np.mean(r)))
