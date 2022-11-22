from color_detect import *

std_color_file = r'E:\code\collor_recorrect\color_value.csv'


def get_A_matrix(x, y):
    """

    : параметр x: входные данные, форма: (10, n)
         : param y: стандартные данные образца, форма: (3, n)
         : return: возвращает обученную матрицу коэффициентов A, форма: (3, 10)
    """
    temp1 = np.dot(x, x.T)
    temp2 = np.dot(x, y.T)
    temp1 = np.linalg.inv(temp1)
    A = np.dot(temp1, temp2)
    return A.T


def get_polynomial(R, G, B):
    """

         : param rgb: RGB значение пикселя, формат (r, g, b)
         : return: Возвращает построенный многочлен, (1, R, G, B, RG, RB, BG, R * R, B * B, G * G)
    """
    R = int(R)
    G = int(G)
    B = int(B)
    return [1, R, G, B, R * G, R * B, B * G, R * R, B * B, G * G]


def create_inputData(image_data):
    """

         : param image_data: исходное изображение для исправления
         : return: возвращает входную матрицу, необходимую для линейной регрессии, форма: (10, image_data.shape [0] * image_data.shape [1])
    """
    data = []
    for raw_data in image_data:
        for bgr in raw_data:
            data.append(get_polynomial(bgr[2], bgr[1], bgr[0]))

    data = np.array(data)

    return data.T


def get_stdColor_value():
    """
         Построить матрицу R, G, B стандартной цветной карты, форма: (3, 24)
         : return: возвращает значения R, G и B стандартной цветовой карты, хранящиеся в словаре и матрице
    """
    color_dict = {}
    std_matrix = []
    color_value_list = np.loadtxt(std_color_file, dtype=np.str, delimiter=',')

    for element in color_value_list:
        color_dict[element[1]] = (int(element[2]), int(element[3]), int(element[4]))
        std_matrix.append([int(element[2]), int(element[3]), int(element[4])])

    std_matrix = np.array(std_matrix)
    return color_dict, std_matrix.T


def recorrect_color(raw_img, A):
    """
         Цветовая коррекция изображения с матрицей коэффициентов A
         : param raw_img: raw image
         : параметр A: матрица коэффициентов
         : return: вернуться к исправленному изображению
    """
    w = raw_img.shape[0]
    h = raw_img.shape[1]
    input_data = create_inputData(raw_img)
    corrected_data = np.dot(A, input_data)
    data = []
    for element in corrected_data:
        vec = []
        for value in element:
            if 0.0 <= value <= 255.0:
                vec.append(int(value))
            elif 0.0 > value:
                vec.append(0)
            elif 255.0 < value:
                vec.append(255)
        data.append(vec)

    data = np.array(data)
    data = data.transpose((1, 0))
    new_img = data.reshape((w, h, 3))
    cv2.imwrite(r'image_without_aberation.jpg.jpg', new_img[..., [2, 1, 0]])
    return new_img


if __name__ == '__main__':
    # Загрузить данные стандартной карты цветов
    color_dict, std_matrix = get_stdColor_value()
    # Загрузите тестовое изображение карты цветов и сгенерируйте входные данные регрессии
    img = cv2.imread(r'image_with_aberation.jpg', 1)
    imgs, color_img = img_split(img)
    input_data = create_inputData(color_img)
    # Рассчитать матрицу коэффициентов уравнения регрессии
    A = get_A_matrix(input_data, std_matrix)
    # Цветовая коррекция
    recorrect_color(img, A)
