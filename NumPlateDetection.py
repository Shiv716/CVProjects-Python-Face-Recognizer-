import cv2 as cv

frameWidth = 480
frameHeight = 640

nPlateCascade = cv.CascadeClassifier("haarCascade_RussianNumberPlate.xml")
minArea = 400
color = (255, 0, 0)

while True:
    img = cv.imread('NumberPlates/plate2.jpeg')
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    nPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)

    for x, y, w, h in nPlates:
        area = w*h
        if area > minArea:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(img, "Number Plate", (x, y-5),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            # Extracting the number plate:-
            imgROI = img[y:y+h, x:x+w]
            cv.imshow('Number plate', imgROI)

    cv.imshow('image', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break