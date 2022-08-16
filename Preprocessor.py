import cv2 as cv
import numpy as np

frameWidth = 480
frameHeight = 640
cap = cv.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)


# Getting the area for shapes:- (edge detection)
def getContours(img):
    biggest = np.array([])
    maxArea = 0

    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        print(area)
        if area > 5000:
            # cv.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            peri = cv.arcLength(cnt, True)  # Arc length of the detected contour.
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)  # Detecting all corner points
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv.drawContours(img_contour, biggest, -1, (255, 0, 0), 20)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int8)
    add = myPoints.sum(1)
    print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPoints[1] = myPoints[np.argmin(diff)]
    myPoints[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def getWarp(img, biggest):
    if biggest != 0:
        biggest = reorder(biggest)

    print(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv.warpPerspective(img, matrix, (frameWidth, frameHeight))

    imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    imgCropped = cv.resize(imgCropped, (frameWidth, frameHeight))

    return imgCropped


def imgProcessing(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDial = cv.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv.erode(imgDial, kernel, iterations=1)

    return imgThres


while True:
    success, img = cap.read()
    cv.resize(img, (frameWidth, frameHeight))

    img_contour = img.copy()
    imgThres = imgProcessing(img)

    biggest = getContours(imgThres)
    if biggest != 0:
        img_warped = getWarp(img, biggest)

    cv.imshow('image', img_warped)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
