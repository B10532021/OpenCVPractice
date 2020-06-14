import cv2
import pytesseract as ocr
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('IDcard.jpg')
# img = cv2.resize(img, (428, 270), cv2.INTER_LINEAR)
print('image size : {}'.format(img.shape))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY_INV) # threshold最好設定在150~180以上
# plt.imshow(binary, cmap='gray')
# plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
dilation = cv2.dilate(binary, kernel)
# plt.imshow(dilation, cmap='gray')
# plt.show()

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area < 2000 and area > 1500:
        ID_contour = contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow('result', img)
cv2.waitKey()


x, y, w, h = cv2.boundingRect(ID_contour)
ID_range = binary[y:y+h, x:x+w]
plt.imshow(ID_range, cmap='gray')
plt.show()

if ID_range is not None:
    image = Image.fromarray(ID_range)
    ocr.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
    result = ocr.image_to_string(image, config=tessdata_dir_config)

    if result:
        print('ID number: {}'.format(result))
    else:
        print('Identify fail')

cv2.destroyAllWindows()