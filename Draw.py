import cv2 as cv
import numpy as num

# Making a blank image.
blank = num.zeros((500, 500, 3), dtype='uint8')
# cv.imshow('blank', blank)

# drawing a blank screen (coloured)
blank[:] = 0, 0, 0
# cv.imshow('Green',blank)

# Drawing a rectangle:
# cv.rectangle(blank, (0, 0), (250,250), (0, 255, 0), thickness=2)
# cv.imshow("Rectangle", blank)

# Drawing a circle:
# cv.circle(blank, (blank.shape[1]//2,blank.shape[0]//2) , 40, (0, 0, 255), thickness=3)
# cv.imshow('Circle', blank)

# Making a line:
# cv.line(blank, (100, 250), (300,400), (255, 255, 255), thickness=3)
# cv.imshow("Line", blank)

# Write text:
cv.putText(blank, 'hello', (170, 255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
cv.imshow('Text', blank)

cv.waitKey(5000)
