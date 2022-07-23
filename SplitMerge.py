import cv2 as cv
import numpy as np

img = cv.imread('gojo.jpeg')
blank = np.zeros(img.shape[:2], dtype='uint8')

b,g,r = cv.split(img)
# cv.imshow('Blue', b)
# and hence for all other colours..

# getting the actual coloured images:
blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])
# cv.imshow('Real blue', blue)

# for getting the value of shapes of each images:
print(b.shape)
print(r.shape)
print (img.shape) # also

# for merging all the colours to get the correct image:
merged_img=cv.merge([b,g,r])
# cv.imshow('merged image', merged_img)

# BLURRING:-
# averaging:
blur = cv.blur(img, (7,7))

# Gaussian blur:
gaussian_blur = cv.GaussianBlur(img, (7,7), 1)

# Median Blur (smudge effect):
median_blur = cv.medianBlur(img, 7)

# bilateral:
bilateral_blur = cv.bilateralFilter(img, 10, 15, 15)

cv.imshow('blurred_image', bilateral_blur)
cv.waitKey(5000)