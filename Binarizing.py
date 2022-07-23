import cv2 as cv
import numpy as np
img = cv.imread('gojo.jpeg')

# gray image:
gray =cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# SIMPLE thresholding
threshold , thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
# inverse of already created threshold:
threshold , thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)

# ADAPTIVE THRESHOLDING:- ( and do-able for inverse threshold value)
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 9)

# GRADIENTS FOR EDGE DETECTION:-
# Laplacian method: (pencil shaded form)
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))

# SOBEL: (edge detection along the axis)
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx,sobely) # combined sobel image

# canny: (most clear form the gradient , multi-stage algorithm)
canny = cv.Canny(gray, 150, 175)

cv.imshow('image',canny)
cv.waitKey(5000)