"""
3.选择1-2种其它特征，重复问题2。
"""
import cv2
import numpy as np
import os

path = os.getcwd()
img_dir_path = path + "\img"
img_list = os.listdir(img_dir_path)
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def get_similarity(des1, des2, threshold=0.75):
    matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
    good = [m for (m, n) in matches if m.distance < threshold * n.distance]
    return len(good) / len(matches)


def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv2.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
    cv2.imshow("pic", outimage)
    cv2.waitKey(0)
    cv2.imwrite("./compare.jpg", outimage)


for i in range(5):
    img_path = img_dir_path + '\\' + img_list[i]
    img = cv2.imread(img_path)
    print("Test img " + str(i) + " : " + img_path)
    img_kp, img_des = orb.detectAndCompute(img, None)
    similarity = []
    result = []
    for j in range(5, 55):
        target_path = img_dir_path + '\\' + img_list[j]
        target_img = cv2.imread(target_path)
        target_kp, target_des = orb.detectAndCompute(target_img, None)
        compare_coefficient = get_similarity(img_des, target_des)
        similarity.append(compare_coefficient)
    top_3_similarity = sorted(similarity)[-3:]
    top_3_similarity.reverse()
    for k in range(len(top_3_similarity)):
        index = similarity.index(top_3_similarity[k])
        while result.__contains__(index):
            index = similarity.index(top_3_similarity[k], index + 1)
        result.append(index)
        print("Similarity rank " + str(k + 1))
        print("Similar image " + img_list[index + 5])
    print("=============================")
img1 = cv2.imread("./img/13.jpg")
img2 = cv2.imread("./img/7.jpg")
img1_kp, img1_des = orb.detectAndCompute(img1, None)
img2_kp, img2_des = orb.detectAndCompute(img2, None)
match = bf.match(img1_des, img2_des)
good_match = []
min_distance = match[0].distance
max_distance = match[0].distance
for x in match:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance
for x in match:
    if x.distance <= max(2 * min_distance, 30):
        good_match.append(x)

draw_match(img1, img2, img1_kp, img2_kp, good_match)
