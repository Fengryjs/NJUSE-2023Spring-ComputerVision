"""
2.在灰度图像I1上增加不同类型（类型>=3）的噪声，分别生成噪声图像。
"""
import cv2
import numpy as np


def gauss(image):
    # 给图片添加高斯噪声
    shape = image.shape
    mean = 0
    sigma = 5
    gauss = np.random.normal(mean, sigma, (shape[0], shape[1]))
    noisy_img = image + gauss
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    return noisy_img


def salt(image):
    # 给图像添加椒盐噪声
    shape = image.shape
    s_vs_p = 0.01
    amount = 0.01

    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    image[coords[0], coords[1]] = 255

    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    image[coords[0], coords[1]] = 0
    return image


def uniform(image):
    # 给图像添加均匀噪声
    shape = image.shape

    mean, sigma = 1, 10
    a = 2 * mean - np.sqrt(12 * sigma)  # a = -14.64
    b = 2 * mean + np.sqrt(12 * sigma)  # b = 54.64
    noise = np.random.uniform(a, b, img.shape)
    noisy_img = image + noise
    noisy_img = np.uint8(cv2.normalize(noisy_img, None, 0, 255, cv2.NORM_MINMAX))
    return noisy_img


img = cv2.imread("./img/hw02-I1.jpeg", cv2.IMREAD_GRAYSCALE)

noisy_img = uniform(img)
# 保存图片
cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
cv2.imshow("pic", noisy_img.astype("uint8"))
cv2.waitKey()
cv2.imwrite("./img/gray-uniform.jpg", noisy_img)

