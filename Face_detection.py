import cv2
import numpy as np
import os

# Setting path for gaining photos of face

directory = r'F:\Python_projects\8. Face_recognition_2\faces'

os.chdir(directory)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)


# Getting photos of face
face_id = int(input("Enter the user id: "))
print("I'm capturing your face. Look on the camera and wait")


count = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (300, 300))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', frame)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite("User." + str(face_id) + '.' + str(count) + '.jpg', gray[y:y+h, x:x+w])
            
    if count >= 200:
        print("Gaining photos completed.")
        break

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()