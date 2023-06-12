"""
3.设计不同类型（类型>=3）的滤波器，对上述噪声图像分别进行去噪。对结果进行分析。
"""
import cv2


def blur_filter(image):
    # 均值滤波
    return cv2.blur(image, (3, 3))


def median_blur_filter(image):
    # 中值滤波
    return cv2.medianBlur(image, 3)


def gauus_blur_filter(image):
    # 高斯滤波
    return cv2.GaussianBlur(image, (3, 3), 1.5)


file_suffix = "gauss"
filter = "gauss"
img = cv2.imread("./img/gray-" + file_suffix + ".jpg", cv2.IMREAD_GRAYSCALE)
filtered = gauus_blur_filter(img)
cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
cv2.imshow("pic", filtered.astype("uint8"))
cv2.waitKey()
cv2.imwrite("./img/gray-" + file_suffix + "-" + filter + ".jpg", filtered)
