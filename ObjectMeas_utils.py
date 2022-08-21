import cv2 as cv
import numpy as np


def getContours(img, cThres=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(img, (5, 5), 1)
    imgCanny = cv.Canny(imgBlur, cThres[0], cThres[1])
    kernel = np.ones((5, 5))
    imgDial = cv.dilate(imgCanny, kernel, iterations=3)
    imgThres = cv.erode(imgDial, kernel, iterations=2)
    if showCanny: cv.imshow('Canny Image', imgThres)

    contours, hierarchy = cv.findContours(imgThres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    finalContours = []
    for i in contours:
        area = cv.contourArea(i)
        if area > minArea:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append(len(approx), area, approx, bbox, i)
            else:
                finalContours.append(len(approx), area, approx, bbox, i)
    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalContours


def reorder(myPoints):
    print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def warpImage(img, points, w, h, pad=20):
    # print(points)
    points = reorder(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]

    return imgWarp


def getDist(pts1, pts2):
    return (pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1] ** 2) ** 0.5
