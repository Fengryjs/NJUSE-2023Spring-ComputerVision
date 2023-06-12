"""
1.将附带的彩色图像（I0）转为灰度图像（记为I1）。
"""
import cv2
import numpy

pic = cv2.imread("img/hw01-I0.jpeg")
row, col, channel = pic.shape
print(row, col, channel)
pic_gray = numpy.zeros((row, col))
print(pic_gray.shape)
for i in range(row):
    for j in range(col):
        pic_gray[i][j] = 1 / 3 * numpy.sum(pic[i][j])
print(pic_gray)
cv2.namedWindow("pic_gray", cv2.WINDOW_NORMAL)
cv2.imshow("pic_gray", pic_gray.astype("uint8"))
cv2.waitKey()
cv2.imwrite("img/hw01-I1.jpeg", img=pic_gray)

