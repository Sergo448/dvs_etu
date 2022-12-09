import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


path_1 = r'//home//sergey//PycharmProjects//dvs_etu//regresor_RGB//3grad_images_plus_rectangles//B_image_l.jpeg'
path_2 = r'//home//sergey//PycharmProjects//dvs_etu//regresor_RGB//3grad_images_plus_rectangles//G_image_r.jpeg'

img_ = cv.imread(path_1)
img1 = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
img = cv.imread(path_2)
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# without error
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

plt.subplot(122)
plt.imshow(dst)
plt.title('Warped Image')
plt.show()

plt.figure()
dst[0: img.shape[0], 0: img.shape[1]] = img
cv.imwrite('result_image.jpg', dst)
plt.imshow(dst)
plt.show()

