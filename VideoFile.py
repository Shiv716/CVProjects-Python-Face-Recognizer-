import cv2 as cv
import caer

# Importing videos:-
# PLAYING A VIDEO
capture = cv.VideoCapture('Movie.mov')
while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
