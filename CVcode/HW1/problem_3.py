"""
3.灰度图像每个像素的灰度值为1个字节（8位），按照从低到高记为L1、L2、…、L8。将I1中每个像素的L1、L2、…、L8分别用I2替换。对结果进行分析。
"""
import cv2
import numpy


def replace(first, second, index):

    # print(bin(first))
    # print(bin(second))
    result = (first & ~(2 ** index)) | (second & (2 ** index))
    # print(bin(result))
    return result

i1 = cv2.imread("img/hw01-I1.jpeg", cv2.IMREAD_GRAYSCALE)
i2 = cv2.imread("img/hw01-I2.png", cv2.IMREAD_GRAYSCALE)
assert i1.shape == i2.shape
shape = i1.shape
print(shape)
cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
i3 = numpy.zeros(shape)
for time in range(8):
    for i in range(shape[0]):
        for j in range(shape[1]):
            i3[i][j] = replace(i1[i][j], i2[i][j], time)
    print("Replace byte " + str(time))
    cv2.imshow("pic", i3)
    cv2.imwrite("./img/hw-replace-" + str(time) + ".png", i3)

