import cv2 as cv

img = cv.imread('grup_people.jpeg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

print(f'Number of faces found {len(faces_rect)}')

# drawing a rectangle around the detected face:
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('person(s)', img)
cv.waitKey(0)
