from weakref import ref
import cv2
import os
from flask import Flask, render_template, Response

from cv2 import FONT_HERSHEY_COMPLEX
# VARIABLES
# measured face width, in cm's
KNOWN_WIDTH = 14.9
# measured distance of face from camera, in cm's
KNOWN_DISTANCE = 31.5
# pretrained classifier to detect faces in frame
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
# for formatting rectangles displayed
WHITE = (255, 255, 255)
STROKE = 2
# paths for reference images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "referenceImages")

#{ function to draw rectangles on detected faces and return width of the face
def getFaceWidth(frame):
    faceWidth = 0
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # convert frame to greyscale for facial detection
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=5)
    # draw rectangles on frame image
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), WHITE, STROKE) 
        faceWidth = w
    return faceWidth

# function to measure the lens' focal length (Similar Triangles)
# this only uses readings from one measured image. To get more accurate, will help to take average of several measurements
def getFocalLength(measuredDistance, realWidth, refImgWidth):
    focalLength = (refImgWidth*measuredDistance) /realWidth
    return focalLength

# function to return the estimated distance of object according to the maesured width
def getDistance(measuredWidth, focalLength,frameWidth):
    distance = (focalLength * measuredWidth) /frameWidth
    return distance


def displayDistanceFrame():
    # for every  reference image from directory
    #for root, dirs, files in os.walk(image_dir): 
    #   for file in files:
    #         path = os.path.join(root, file)
    #         print(path)
    imgPath = os.path.join(os.path.dirname(__file__), 'referenceImages/3.jpg')
    refImage = cv2.imread(imgPath)

    # train model to detect pixels for known width, and find focal length
    refFaceWidth = getFaceWidth(refImage)
    focalLength = getFocalLength(KNOWN_DISTANCE, KNOWN_WIDTH,refFaceWidth)
    print(focalLength)

    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        # ensure a frame is read
        if not success:
            break
        # measure current width and calcualte distance
        measuredWidth = getFaceWidth(frame)
        if measuredWidth != 0:
            distance = getDistance(KNOWN_WIDTH, focalLength, measuredWidth)
        # add text to live stream
        cv2.putText(   
            frame, f"Distance = {round(distance, 2)} cm", (50,50), FONT_HERSHEY_COMPLEX, 1, WHITE, STROKE
        )
       
        # image encode makes image format into streaming data for network transmission
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        if cv2.waitKey(1) == ord('q'):
            break
            
    camera.release() 
    cv2.destroyAllWindows()        
