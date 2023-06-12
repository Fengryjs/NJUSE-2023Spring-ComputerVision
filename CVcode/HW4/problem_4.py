import cv2
import numpy
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # xml来源于资源文件。
exit_img = cv2.imread("./exit_frame.jpg")
exit_gray = cv2.cvtColor(exit_img, cv2.COLOR_BGR2GRAY)
face_data = []
labels = []
id = 0
faces = face_cascade.detectMultiScale(exit_gray)
for (x, y, w, h) in faces:
    face_data.append(exit_gray[y:y + h, x: x + w])
    labels.append(id)
    cv2.rectangle(exit_img, (x, y), (x + w, y + h),
                  (0, 0, 255), 5)
    cv2.putText(exit_img, str(id), (x, y), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 0, 0), 5)
    id += 1
recognizer.train(face_data, numpy.array(labels))
test_img = cv2.imread("./re_enter_frame.jpg")
test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(test_gray)
for (x, y, w, h) in faces:
    id, confidence = recognizer.predict(test_gray[y:y+h, x:x+w])
    cv2.rectangle(test_img, (x, y), (x + w, y + h),
                  (0, 0, 255), 5)
    cv2.putText(test_img, str(id) + ", " + str(confidence), (x, y), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 0, 0), 5)
exit_img = cv2.resize(exit_img, dsize=None, fx=0.3, fy=0.3)
test_img = cv2.resize(test_img, dsize=None, fx=0.3, fy=0.3)
cv2.imwrite("./problem4/data.jpg", exit_img)
cv2.imwrite("./problem4/result.jpg", test_img)
cv2.imshow("a", exit_img)
cv2.waitKey(0)
cv2.imshow("a", test_img)
cv2.waitKey(0)