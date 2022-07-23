import cv2 as cv
import numpy as np
import os

people = []
DIR = r'/Users/shivangchaudhary/Documents/OpenCV-ImgFolders'

for image in os.listdir(DIR):
    people.append(image)

print(people)

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

features = []
labels = []


def createTrain():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for image in os.listdir(path):
            img_path = os.path.join(path, image)
            img_array = cv.imread(img_path)
            # if img_array is not None:
            # print(image)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            # print(gray)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                labels.append(label)
                features.append(faces_roi)


createTrain()
print('Training done ------------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recogniser = cv.face.LBPHFaceRecognizer_create()

# Training the face recogniser in features and labels list:
face_recogniser.train(features, labels)

face_recogniser.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
