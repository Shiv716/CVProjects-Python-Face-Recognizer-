import cv2 as cv


# TO re-shape the frame size:-
def reScaleFrame(frame, scale=0.2):
    # Will work for images , videos and live-videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def resChange(width , height):
    # Will work for live videos:-
    capture.set(3, width)
    capture.set(5, height)


capture = cv.VideoCapture('Movie.mov')

while True:
    isTrue, frame = capture.read()

    frame_resized = reScaleFrame(frame)

    # Checking the original dimension of the video:
    cv.imshow('Original video', frame)

    cv.imshow('Video-resized', frame_resized)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
