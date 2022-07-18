import cv2 as cv
import caer

# Will read the image
image = cv.imread('gojo.jpeg', 1)
# Will load the image:-
cv.imshow('image', image)
cv.waitKey(5000)
cv.destroyAllWindows()

# loading the facial data from opencv:- (not taking in parameters)

trained_face_data = cv.CascadeClassifier()

# Gray Image Conversion:
grayscaled_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# Detect Faces:- (not working)
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)

# (x,y,w,h) = face_coordinates[0] (CAN'T WORK UNTIL FACE-COORDINATES DOES..)
# cv.rectangle(image,(x+w,y+h),(x,y),(0,255,0),2)


# Importing videos:-
# GET A VIDEO TO START WITH;

# TO re-shape the frame size:-
def reScaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)