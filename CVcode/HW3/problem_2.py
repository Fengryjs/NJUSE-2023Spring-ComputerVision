"""
2.以全局RGB颜色直方图（每通道bin的数量为8）作为特征，进行图像检索。展示每个检索请求及对应的前3个结果。
"""
import cv2
import sys
import os


def get_hist_feature(image):
    image_hist = cv2.calcHist([image], [i for i in range(img.shape[2])], None, [8 for i in range(img.shape[2])],
                            [0, 256, 0, 256, 0, 256])
    image_hist = cv2.normalize(image_hist, image_hist, 0, 255, cv2.NORM_MINMAX).flatten()
    return image_hist


path = os.getcwd()
img_dir_path = path + "\img"
img_list = os.listdir(img_dir_path)
assert len(img_list) == 55

for i in range(5):
    img_path = img_dir_path + '\\' + img_list[i]
    print("Test img " + str(i) + " : " + img_path)
    img = cv2.imread(img_path)
    img_hist = get_hist_feature(image=img)
    similarity = []

    for j in range(5, 55):
        target_path = img_dir_path + '\\' + img_list[j]
        target_img = cv2.imread(target_path)
        target_hist = get_hist_feature(image=target_img)
        compare_coefficient = cv2.compareHist(img_hist, target_hist, cv2.HISTCMP_INTERSECT)
        similarity.append(compare_coefficient)
    top_3_similarity = sorted(similarity)[-3:]
    top_3_similarity.reverse()
    for k in range(len(top_3_similarity)):
        index = similarity.index(top_3_similarity[k])
        print("Similarity rank " + str(k + 1))
        print("Similar image " + img_list[index + 5])
    print("=============================")
