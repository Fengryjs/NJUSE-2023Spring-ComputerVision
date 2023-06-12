"""
4.将附带彩色图像I0的R、G、B通道中某个或某几个通道做与问题3类似的处理。对结果进行分析。
"""

import cv2
import numpy


def replace(first, second, index):
    return (first & ~(2 ** index)) | (second & (2 ** index))


i0 = cv2.imread("img/hw01-I0.jpeg")
i2 = cv2.imread("img/hw01-I2.png", cv2.IMREAD_GRAYSCALE)
shape = i0.shape
print(shape)
cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
i3 = numpy.copy(i0)
for time in range(8):
    for i in range(shape[0]):
        for j in range(shape[1]):
            # Red channel
            i3[i][j][0] = replace(i0[i][j][0], i2[i][j], time)
            # Green channel
            i3[i][j][1] = replace(i0[i][j][1], i2[i][j], time)
            # Blue channel
            i3[i][j][2] = replace(i0[i][j][2], i2[i][j], time)
    print("Replace byte " + str(time))
    cv2.imshow("pic", i3)
    cv2.imwrite("./img/hw-all-replace-" + str(time) + ".png", i3)