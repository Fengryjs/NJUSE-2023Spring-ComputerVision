import numpy as np
import cv2
frame_to_record = [30 * 3 * i for i in range(10)]
re_enter_frame = 660
exit_frame = 150
print(frame_to_record)
video = cv2.VideoCapture("./test.mp4")
index = 0
while True:
    # 获取一帧
    ret, frame = video.read()  # 读取成功后 ret 返回为为布尔值True,frame返回读取的一帧图像
    if not ret:
        break
    if index in frame_to_record:
        cv2.imwrite("./problem1/video_frame_" + str(frame_to_record.index(index)) + ".jpg", frame)
    if index == re_enter_frame:
        cv2.imwrite("./re_enter_frame.jpg", frame)
    if index == exit_frame:
        cv2.imwrite("./exit_frame.jpg", frame)
    index += 1
print(index)
