import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(images, labels), _ = mnist.load_data() 
train_images = np.array(images[:59990, :])
train_images = train_images.reshape(-1, 784).astype(np.float32)
train_labels = labels[:59990, np.newaxis].astype(np.int32)
print('train images shape:', train_images.shape)
print('train labels shape:', train_labels.shape)

test_images = np.array(images[59990:, :])
test_images = test_images.reshape(-1, 784).astype(np.float32)
test_labels = labels[59990:, np.newaxis].astype(np.int32)
print('test images shape:', test_images.shape)
print('test labels shape:', test_labels.shape)

# 分析特徵影像的特徵值
knn = cv2.ml.KNearest_create()
knn.train(train_images, cv2.ml.ROW_SAMPLE, train_labels)
# ret, result, neighbours, dist = knn.findNearest(test_images, k = 5)

# print('test   label:', test_labels.ravel())
# print('result label:', result.astype(np.int32).ravel())

# 識別連續數字的照片
img = cv2.imread('8342.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
blur = cv2.blur(binary, (5, 5))
kernal = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(blur ,cv2.MORPH_OPEN, kernal) # 先erode再dilate
plt.imshow(opening, cmap='gray')
plt.show()

contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img, str(i), tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

cv2.imshow('result', img)
cv2.waitKey()
cv2.destroyAllWindows()

results = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 200:
        x, y, w, h = cv2.boundingRect(contour)
        print(x)
        num = opening[y:y+h, x:x+w]
        rect = cv2.resize(num, images[0].shape).ravel()
        _, res, _, _ = knn.findNearest(np.array([rect]).astype(np.float32), k = 5)
        results.append(int(res.ravel()[0]))

results.reverse()
print(results)