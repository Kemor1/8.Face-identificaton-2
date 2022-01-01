import os
import cv2
import numpy as np
import os


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('F:\\Python_projects\\8. Face_recognition_2\\trainer\\trainer.yml')
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['None', 'Tom']


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (300, 300))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('image', frame)

    faces = detector.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        if confidence < 100:
            id = names[id]
            confidence = f"{round(100 - confidence)}%"
        else:
            id = 'unknown'
            confidence = f"{round(100 - confidence)}%"
        
        cv2.putText(frame, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', frame)

        
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()