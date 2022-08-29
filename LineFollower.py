import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
hsvVals = [0, 0, 117, 179, 22, 219]

sensors = 3
threshold = 0.2
width, height = 480, 360
sensitivity = 3 # if number is high = less sensitive

weights = [-25, -25, 0, 15, 25]
CURVE = 0


def thresholding(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv.inRange(hsv, lower, upper)
    return mask


def getContours(imgThres, img):
    cx = 0
    contours, hierarchy = cv.findContours(imgThres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        biggest = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(biggest)
        cx = x + w//2
        cy = y + w//2
        cv.drawContours(img, biggest, -1, (255, 0, 255), 5)
        cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)
    return cx


def getSensorOutput(imgThres, sensors):
    imgs = np.hsplit(imgThres, sensors)
    totalPixels = (img.shape[1]//sensors) * img.shape[0]
    senOut = []
    for x, im in enumerate(imgs):
        pixelCount = cv.countNonZero(im)
        if pixelCount > threshold * pixelCount:
            senOut.append(1)
        else:
            senOut.append(0)
        # cv.imshow(str(x), im)
    # print(senOut)
    return senOut


def sendCommands(senOut, cx):
    global curve

    # TRANSLATION:-
    lr = (cx - width // 2) // sensitivity
    lr = int (np.clip(lr, -10, 10))
    if lr > -2 and lr < 2: lr= 0

    # ROTATION:-
    if senOut == [1, 0, 0]: curve = weights[0]
    if senOut == [1, 1, 0]: curve = weights[1]
    if senOut == [0, 1, 0]: curve = weights[2]
    if senOut == [0, 1, 1]: curve = weights[3]
    if senOut == [0, 0, 1]: curve = weights[4]
    # DEAD SITUATIONS:-
    if senOut == [0, 0, 0]: curve = weights[2]
    if senOut == [1, 1, 1]: curve = weights[2]
    if senOut == [1, 0, 1]: curve = weights[2]


while True:
    _, img = cap.read()
    img = cv.resize(img, (width, height))
    img = cv.flip(img, 0)

    imgThres = thresholding(img)
    # FOR TRANSLATION:-
    cx = getContours(imgThres, img)
    # FOR ROTATION:-
    senOut = getSensorOutput(imgThres, sensors)

    sendCommands(senOut, cx)

    cv.imshow('Image', img)
    cv.imshow('Image-Thres', imgThres)
    cv.waitKey(1)
