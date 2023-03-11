import cv2
import numpy as np

# img = cv2.imread("./RunningYOLO/Images/bikes.jpeg")

zeros = np.zeros((400, 400), dtype='uint8')
img = cv2.merge([zeros, zeros, 255])
print(img.shape)
cv2.imshow("Image", img)

x1, y1 = 0, 400
x2, y2 = 200, 400
line = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 15)

cv2.waitKey(0)
