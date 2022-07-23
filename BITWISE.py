import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as pt

img = cv.imread('gojo.jpeg')

blank = np.zeros((400,400), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

# cv.imshow('Rectangle', rectangle)
# cv.imshow('Circle', circle)

# bitwise AND: (intersecting regions)
bitwise_and = cv.bitwise_and(rectangle, circle)

# bitwise OR: (complete portion)
bitwise_or = cv.bitwise_or(rectangle,circle)

# bitwise XOR : (non-intersecting region)
bitwise_xor = cv.bitwise_xor(rectangle,circle)

# bitwise NOT: (complement or inversion of the already exisiting image)
bitwise_not = cv.bitwise_not(circle)

# MASKING::
blank = np.zeros(img.shape[:2], dtype='uint8')
mask = cv.circle(blank , (img.shape[1]//2 , img.shape[0]//2), 300 , 255,-1)

# finally,
masked = cv.bitwise_and(img,img,mask=mask)

# HISTOGRAMS:
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# grayscale histogram,
gray_hist = cv.calcHist([gray], [0] , mask , [256], [0,256])

# plotting using pyplot
# plt.figure()
# plt.title('GrayScale Histogram')
# plt.xlabel('bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

# colour histogram:
plt.figure()
plt.title('Colour histogram')
plt.xlabel('bins')
plt.ylabel('# of bins')
colours = ('b', 'g', 'r')
for i,col in enumerate(colours):
    hist = cv.calcHist([img], [i], mask, [256], [0,256])
    plt.plot(hist, color = col)
    plt.xlim([0,256])
plt.show()

# cv.imshow('Image', masked)
# cv.waitKey(5000)