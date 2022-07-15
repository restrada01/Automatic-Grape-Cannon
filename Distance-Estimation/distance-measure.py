import cv2

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

# function to draw rectangles on detected faces and return width of the face
def getFaceData(frame):
    face_width = 0
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # convert frame to greyscale for facial detection
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=5)
    # draw rectangles on frame image
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), WHITE, STROKE) 
        face_width = w
    return face_width


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    detectedFaceWidth = getFaceData(frame)
    print(detectedFaceWidth)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()