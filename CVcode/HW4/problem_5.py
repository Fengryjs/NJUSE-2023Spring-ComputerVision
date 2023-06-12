import numpy as np
import cv2

frame_to_record = [30 * 3 * i for i in range(10)]
re_enter = 660
# 实例化人脸分类器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # xml来源于资源文件。
# 读取测试图片
video = cv2.VideoCapture("./test.mp4")
fps = int(video.get(cv2.CAP_PROP_FPS))
video_height, video_width = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter('./result.mp4', fourcc, fps, (video_width, video_height), True)
# params
trackers = None
train_data = []
labels = []
id = 0
index = 0
recognizer = cv2.face.LBPHFaceRecognizer_create()
threhold = 70
while True:
    # 获取一帧
    ret, frame = video.read()  # 读取成功后 ret 返回为为布尔值True,frame返回读取的一帧图像
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (height, width) = frame.shape[:2]
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 前面图像识别中注释过了
    if index == 0:
        if len(faces) > 0:
            trackers = []
            for (x, y, w, h) in faces:
                tracker = cv2.legacy.TrackerMIL_create()
                tracker.init(frame, (x, y, w, h))
                trackers.append(tracker)
                train_data.append(gray[y:y + h, x: x + w])
                labels.append(id)
                id += 1
    recognizer.train(train_data, np.array(labels))
    tracker_to_remove_list = []

    if trackers is not None:
        # grab the new bounding box coordinates of the object
        for i in range(len(trackers)):
            (success, box) = trackers[i].update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                # green
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 5)
                target_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                if confidence > threhold:
                    print("remove tracker")
                    tracker_to_remove_list.append(trackers[i])
                cv2.putText(frame, str(target_id) + ", " + str(confidence), (x, y), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 0), 5)
            else:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for item in tracker_to_remove_list:
        trackers.remove(item)
    for (x, y, w, h) in faces:
        target_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        # 新的人脸，不在数据集中所以置信度指数很高，不同的recognizer这个指数的比较方式不同
        if confidence > threhold:
            print("Add new data")
            # 优化分类器和train_data
            tracker = cv2.legacy.TrackerMIL_create()
            tracker.init(frame, (x, y, w, h))
            trackers.append(tracker)
            train_data.append(gray[y:y + h, x: x + w])
            labels.append(id)
            id += 1
            recognizer.train(train_data, np.array(labels))
        # 已有人脸但是没有跟踪的对象
        exist = False
        for tracker in trackers:
            (success, box) = tracker.update(frame)
            if success:
                (track_x, track_y, track_w, track_h) = [int(v) for v in box]
                track_id, track_confidence = recognizer.predict(gray[track_y: track_y + track_h, track_x:track_x + track_w])
                if target_id == track_id and confidence < threhold and track_confidence < threhold:
                    exist = True
        if not exist:
            tracker = cv2.legacy.TrackerMIL_create()
            tracker.init(frame, (x, y, w, h))
            trackers.append(tracker)
    videoWriter.write(frame)
    frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)
    cv2.imshow('faces', frame)
    if cv2.waitKey(1) == ord('b'):  # 按下‘b’键退出窗口
        break
    index += 1
    print(index)
videoWriter.release()
video.release()