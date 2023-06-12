import numpy as np
import cv2
frame_to_record = [30 * 3 * i for i in range(10)]
print(frame_to_record)
# 实例化人脸分类器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # xml来源于资源文件。

video = cv2.VideoCapture("./test.mp4")
index = 0
sum = 0
while True:
    # 获取一帧
    ret, frame = video.read()  # 读取成功后 ret 返回为为布尔值True,frame返回读取的一帧图像
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)
    if index in frame_to_record:
        cv2.imwrite("./problem2/video_frame_" + str(frame_to_record.index(index)) + ".jpg", frame)
    cv2.imshow('faces', frame)
    print(index, len(faces))
    index += 1
    if cv2.waitKey(1) == ord('b'):  # 按下‘b’键退出窗口
        break
print(index)