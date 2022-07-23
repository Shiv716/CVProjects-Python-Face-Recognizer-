import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('opencv.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 125, 175)

# thresholds:
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

# Contours:
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

# binary format:
blank = np.zeros(img.shape, dtype='uint8')

# drawing contours:
cv.drawContours(blank, contours, -1, (0, 0, 255), 2)

# spaces:-
# BGR TO HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# BGR TO L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# bgr to rgb for avoiding inversion of colours in image:
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# converting hsv and lab back to bgr:
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

# for printing:-
cv.imshow('image', lab_bgr)
cv.waitKey(5000)
cv.destroyAllWindows()

# printing using matplotlib:
#plt.imshow(rgb)
#plt.show()
