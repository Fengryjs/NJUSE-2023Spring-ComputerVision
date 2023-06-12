"""
4.尝试对彩色图像I0添加噪声，并设计滤波器进行去噪。对结果进行分析。
"""
import cv2
import numpy as np

image = cv2.imread("./img/hw02-I0.jpg")
shape = image.shape
s_vs_p = 0.01
amount = 0.01

num_salt = np.ceil(amount * image.size * s_vs_p)
coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
image[coords[0], coords[1], :] = [255, 255, 255]

num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
image[coords[0], coords[1], :] = [0, 0, 0]
cv2.imwrite("./img/rgb-salt.jpg", image)

image = cv2.medianBlur(image, 5)
cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
cv2.imshow("pic", image.astype("uint8"))
cv2.waitKey()
cv2.imwrite("./img/rgb-filtered.jpg", image)
