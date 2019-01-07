import cv2
import numpy as np

cv2.setUseOptimized(True)
cv2.setNumThreads(cv2.getNumberOfCPUs())

rgb_image = cv2.imread('cube.jpg')
(l, a, b) = cv2.split(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2Lab))
cv2.imshow('Original image', rgb_image)
cv2.moveWindow('Original image', 0, 0)
lab = np.hstack((l, a, b))
cv2.imshow('Colour components of Lab', lab)
cv2.moveWindow('Colour components of Lab', 400, 0)
cv2.waitKey()
cv2.destroyAllWindows()
