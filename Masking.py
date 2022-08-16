import cv2
import cv2 as cv
import numpy as np

def empty():
    pass


cv.namedWindow('Trackbars')
cv.resizeWindow('Trackbars', 640, 240)
cv.createTrackbar('Hue min', 'Trackbars', 19, 179, empty)
cv.createTrackbar('Hue max', 'Trackbars', 179, 179, empty)
cv.createTrackbar('Sat min', 'Trackbars', 45, 255, empty)
cv.createTrackbar('Sat max', 'Trackbars', 255, 255, empty)
cv.createTrackbar('Val min', 'Trackbars', 20, 255, empty)
cv.createTrackbar('Val max', 'Trackbars', 255, 255, empty)

while True:
    image = cv.imread('gojo.jpeg')

    h_min = cv.getTrackbarPos('Hue min', 'Trackbars')
    h_max = cv.getTrackbarPos('Hue max', 'Trackbars')
    s_min = cv.getTrackbarPos('Sat min', 'Trackbars')
    s_max = cv.getTrackbarPos('Sat max', 'Trackbars')
    v_min = cv.getTrackbarPos('Val min', 'Trackbars')
    v_max = cv.getTrackbarPos('Val max', 'Trackbars')

    print(h_min, h_max, s_min, s_max, v_min, v_max)
    hsvImage = cv.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Masking the images:-
    lower_array = np.array([h_min, s_min, v_min])
    higher_array = np.array([h_max, s_max, v_max])
    mask = cv.inRange(hsvImage, lower_array, higher_array)
    img_result= cv.bitwise_and(image,image,mask)

    cv.imshow('image',image)
    cv.imshow('hsv',hsvImage)
    cv.imshow('mask',mask)
    cv.imshow('final-image',img_result)
    cv.waitKey(1)

