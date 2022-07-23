import cv2 as cv

# colour changing:-
img = cv.imread('gojo.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# blur:
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)

# edge cascade:
canny = cv.Canny(img, 125, 175)

# dilating:
dilated = cv.dilate(canny, (3, 3), iterations=3)

# eroding:
eroded = cv.erode(dilated, (7, 7), iterations=3)

# resizing:
resized_img = cv.resize(eroded, (500, 500), interpolation=cv.INTER_AREA)

# cropping:
cropped_img = img[50:200, 300:400]

# for printing:-
cv.imshow('Image', cropped_img)
cv.waitKey(5000)
cv.destroyAllWindows()