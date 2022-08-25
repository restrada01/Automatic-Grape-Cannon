import os
import cv2

import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# used to number each new person's label and corresponding data
current_id = 0
label_ids = {}
y_labels = []
x_train = []

# walk through directory and find all images to put into training list
for root, dirs, files in os.walk(image_dir): 
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # number each label and add to dict
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            # convert images into NUMPY arrays for pixel values
            pil_image = Image.open(path).convert("L") # convert to grayscale
            image_array = np.array(pil_image, "uint8")
            
            # detect faces on training data, add to training set and with correspoinding label
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi) 
                y_labels.append(id_)           

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")