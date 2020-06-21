import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('faces.jpg')
image = cv2.resize(image, (1280, 960))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = faceCascade.detectMultiScale(
    gray, 
    scaleFactor = 1.15, 
    minNeighbors = 3,
    minSize = (5, 5)
    )

print('發現{}個人臉'.format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imshow("detect", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('results.jpg', image)
