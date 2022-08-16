import cv2 as cv
import numpy as np


# Getting the area for shapes:- (edge detection)
def getContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        print(area)
        if area > 500:
            cv.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            peri = cv.arcLength(cnt, True)  # Arc length of the detected contour.
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)  # Detecting all corner points
            print(approx)
            # exact co-ordinates:-
            Corners = len(approx)
            # Getting the intial and final points of shapes:-
            x, y, w, h = cv.boundingRect(approx)

            # Drawing the rectangle:-
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Writing the names of the shapes:-
            if Corners == 3:
                objType = 'Triangle'
            elif Corners == 4:
                aspRatio = w / float(h)
                if 0.95 < aspRatio < 1.05:
                    objType = "Square"
                else:
                    objType = "Rectangle"
            else:
                objType = "None"
            # Putting the text:-
            cv.putText(img_contour, objType, (x + (w // 2) - 10, y + (h // 2) - 10), cv.FONT_HERSHEY_COMPLEX, 0.5,
                       (0, 0, 0), 1)


img = cv.imread('shapes.png')
# Copying image:-
img_contour = img.copy()

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_Blur = cv.GaussianBlur(img_gray, (7, 7), 1)
img_canny = cv.Canny(img_Blur, 50, 50)
getContours(img_canny)

while True:
    cv.imshow('contour-image', img_contour)
    # cv.imshow('image', img)

    cv.waitKey(2000)
