import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lena.png')
print(img.shape)

data = img.reshape(-1, 3).astype(np.float32)

K = 2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
result = np.array([centers[i] for i in labels.flatten()])
result = result.reshape((img.shape))

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result.png', result)