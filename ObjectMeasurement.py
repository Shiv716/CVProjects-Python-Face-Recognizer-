import cv2 as cv
import numpy as np
import ObjectMeas_utils as utlis

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 160)

while True:
    success, img = cap.read()
    img, conts = utlis.getContours(img, minArea=50000, filter=4, draw=True)

    if len(conts) != 0:
        biggest = conts[0][2]
        # print(biggest)
        imgWarp = utlis.warpImage(img, biggest, 210, 297)
        imgConts2, conts2 = utlis.getContours(imgWarp, minArea=2000, filter=4, draw=False, cThres=[50, 50])

        if len(conts) != 0:
            for obj in conts2:
                cv.polylines(imgConts2, [obj[2]], True, (0, 255, 0), 2)
                npoints = utlis.reorder(obj[2])
                nW = round(utlis.getDist(npoints[0][0]//3, npoints[1][0]//3) / 10, 1)
                nH = round(utlis.getDist(npoints[0][0]//3, npoints[2][0]//3) / 10, 1)
                cv.arrowedLine(imgConts2, (npoints[0][0][0], npoints[0][0][1]),
                                (npoints[1][0][0], npoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv.arrowedLine(imgConts2, (npoints[0][0][0], npoints[0][0][1]),
                                (npoints[2][0][0], npoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv.putText(imgConts2, '{}cm'.format(nW), (x + 30, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv.putText(imgConts2, '{}cm'.format(nH), (x - 70, y + h // 2), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)

        cv.imshow('WarpedImg', imgConts2)

    cv.imshow('Original-Image', img)
    cv.waitKey(1)