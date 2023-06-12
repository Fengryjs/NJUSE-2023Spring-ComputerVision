import numpy as np
import cv2

frame_to_record = [30 * 3 * i for i in range(10)]
re_enter = 660
# 实例化人脸分类器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # xml来源于资源文件。
# 读取测试图片
video = cv2.VideoCapture("./test.mp4")
# params
trackers = None
index = 0
while True:
    # 获取一帧
    ret, frame = video.read()  # 读取成功后 ret 返回为为布尔值True,frame返回读取的一帧图像
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (height, width) = frame.shape[:2]

    if index == 0:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 前面图像识别中注释过了
        if len(faces) > 0:
            cv2.imwrite("./test_face.jpg", frame)
            trackers = []
            for (x, y, w, h) in faces:
                tracker = cv2.legacy.TrackerMIL_create()
                tracker.init(frame, (x, y, w, h))
                trackers.append(tracker)
    if index == re_enter:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        assert len(faces) == 2
        trackers = []
        for (x, y, w, h) in faces:
            tracker = cv2.legacy.TrackerMIL_create()
            tracker.init(frame, (x, y, w, h))
            trackers.append(tracker)
    if trackers is not None:
        # grab the new bounding box coordinates of the object
        for i in range(len(trackers)):
            (success, box) = trackers[i].update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 5)
            else:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)
    if index in frame_to_record:
        cv2.imwrite("./problem3/video_frame_" + str(frame_to_record.index(index)) + ".jpg", frame)
    cv2.imshow('faces', frame)
    if cv2.waitKey(1) == ord('b'):  # 按下‘b’键退出窗口
        break
    index += 1
    print(index)
