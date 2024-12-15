import cv2
import numpy as np
import face_recognition
import os
import sys

# Function to get the absolute path to a resource
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Specify the path to the folder containing the images
image_path = resource_path('C:/Users/Suman Uppala/Desktop/open cv/face_recog/images')

# Initialize lists for storing images and class names
images = []
classNames = []

# Get the list of image files in the specified folder
myList = os.listdir(image_path)
print(myList)

# Loop through the image files and load them
for cl in myList:
    curImg = cv2.imread(f'{image_path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    print(classNames)

# Function to find the encodings of the images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Check if any encodings are found
            encodeList.append(encodings[0])
        else:
            print("No face found in image")
    return encodeList

# Find encodings for the known images
encodeListKnown = findEncodings(images)
print('Encoding complete')

# Specify the path to the image you want to process
test_image_path = resource_path('C:/Users/Suman Uppala/Desktop/open cv/face_recog/images/gill.jpg')

# Load and process the image
img = cv2.imread(test_image_path)
imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

# Find face locations and encodings in the image
facesCurFrame = face_recognition.face_locations(imgS)
encodesCurframe = face_recognition.face_encodings(imgS, facesCurFrame)

# Loop through the detected faces and find matches
for encodeFace, faceLoc in zip(encodesCurframe, facesCurFrame):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    print(faceDis)
    matchIndex = np.argmin(faceDis)

    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
        print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

# Display the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
