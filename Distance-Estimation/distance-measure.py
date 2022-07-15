import cv2
import os

# VARIABLES
# measured face width, in cm's
KNOWN_WIDTH = 16.3
# measured distance of face from camera, in cm's
KNOWN_DISTANCE = 105
# pretrained classifier to detect faces in frame
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
# for formatting rectangles displayed
WHITE = (255, 255, 255)
STROKE = 2
# paths for reference images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "referenceImages")

# function to draw rectangles on detected faces and return width of the face
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
def getFocalLength(measuredWidth):
    focalLength = (measuredWidth * KNOWN_DISTANCE) / KNOWN_WIDTH
    return focalLength

# function to return the estimated distance of object according to the maesured width
def getDistance(measuredWidth, focalLength):
    distance = (focalLength * KNOWN_WIDTH) / measuredWidth
    return distance

# for every  reference image from directory
#for root, dirs, files in os.walk(image_dir): 
 #   for file in files:
  #         path = os.path.join(root, file)
   #         print(path)
imgPath = os.path.join(os.path.dirname(__file__), 'referenceImages/5.jpg')
refImage = cv2.imread(imgPath)

# train model to detect pixels for known width, and find focal length
refFaceWidth = getFaceWidth(refImage)
focalLength = getFocalLength(refFaceWidth)
print(focalLength)
refImageRS = cv2.resize(refImage, (960, 540))
cv2.imshow("Ref Image", refImageRS)


cap = cv2.VideoCapture(0)
while True:
    #ret, frame = cap.read()

   # measuredWidth = getFaceWidth(fra
   if cv2.waitKey(1) == ord('q'):
       break

#cap.release() 
cv2.destroyAllWindows()