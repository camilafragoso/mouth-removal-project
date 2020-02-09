import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.util import random_noise

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

face_cascade = cv2.CascadeClassifier('class/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('class/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('class/haarcascade_smile.xml')

while True:
    ret, frame = cap.read()
    fundo = frame.copy()
    cinza = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = cinza.copy()
    faces = face_cascade.detectMultiScale(cinza, scaleFactor=1.3, minNeighbors=5)


    for (a, b, c, d) in faces:
        face = cinza[b:b+d, a:a+c]
        bocas = smile_cascade.detectMultiScale(face, 1.9, 20)
        for (x, y, w, h) in bocas:
            mask[0:mask.shape[0], 0:mask.shape[1]] = 0
            mask[y+b:y+b+h, x+a:x+a+w] = 255
            frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)


    cv2.imshow('Input1', mask)
    cv2.imshow('Input2', frame)

    c = cv2.waitKey(10)
    if c == 27:
        break

#cv2.imshow('Input2', frame)
cap.release()
cv2.destroyAllWindows()
