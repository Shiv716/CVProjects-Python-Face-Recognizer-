import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

people = ['JerrySienfield', 'BenAffleck', 'Madonna', 'EltonJohn']

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy', allow_pickle=True)

face_recogniser = cv.face.LBPHFaceRecognizer_create()
face_recogniser.read('face_trained.yml')

img = cv.imread('/Users/shivangchaudhary/Documents/OpenCV-ImgFolders/Madonna/Madonna_2.webp')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Person-gray', img)

# Detect face in the image
face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in face_rect:
    face_roi = gray[y:y+h , x:x+w]

    label, confidence = face_recogniser.predict(face_roi)
    print(f'Label {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)

    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Face Detected', img)
cv.waitKey(0)

