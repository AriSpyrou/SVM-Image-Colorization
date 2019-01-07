import cv2
import numpy as np

# Use multiple threads where possible
cv2.setUseOptimized(True)
cv2.setNumThreads(cv2.getNumberOfCPUs())

# Read the image and convert it to Lab
im = cv2.imread('castle.jpg')
# TODO resize if needed
im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
(l, a, b) = cv2.split(im)
del l

# Make a list with a, b coordinates
ab = np.zeros((a.shape[0]*a.shape[1], 2), int)
ab_ind = 0
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        ab[ab_ind] = [a[i][j], b[i][j]]
        ab_ind += 1
del a, b, im, ab_ind

median = np.median(ab, axis=0, overwrite_input=True)
ab = ab[ab[:, 0].argsort()]
for bin in ab