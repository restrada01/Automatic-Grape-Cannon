import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")  # read the trained facial recognizer

# load pickle with dict of labels (person's name corresponding to number)
labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}     # it loads as {"person": #} and we need to use number to get name

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # convert to gray
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # draw rectangles on frame image
    for (x, y, w, h) in faces: 
        roi_gray = gray[y: y+h, x:x+w]      # get region of interest from webcam frame
        roi_color = frame[y: y+h, x:x+w] 

        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf <= 90:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (235,255, 255)
            thickness = 1
            cv2.putText(frame, name, (x, y), font, 1, color, thickness, cv2.LINE_AA)

        #draw rectangle around face
        color = (65, 201, 235)  # RGB value
        thickness = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)            
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()