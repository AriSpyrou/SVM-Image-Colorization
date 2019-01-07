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
        ab[ab_ind] = [a[i, j], b[i, j]]
        ab_ind += 1
del a, b, im, ab_ind

# median = np.median(ab, axis=0, overwrite_input=True)
# ab = ab[ab[:, 0].argsort()]
# Points at which the pixels are divided
bin_list = [[0, ab.shape[0]]]
# Channel is 0 for alpha and 1 for beta
channel = 0

for x in range(5):
    for y in range(2**x):
        median = np.median(ab[bin_list[2**x - 1 + y][0]:bin_list[2**x - 1 + y][1]], axis=0)
        # TODO a[a[:, 0].argsort()[2:5]]
        ab[bin_list[2**x - 1 + y][0]:bin_list[2**x - 1 + y][1]] = ab[ab[bin_list[2**x - 1 + y][0]:bin_list[2**x - 1 + y][1], channel].argsort()]
        for i in range(bin_list[2**x - 1 + y][0], ab.shape[0]):
            if ab[i][channel] >= median[channel]:
                bin_list.append([bin_list[2**x - 1 + y][0], i])
                bin_list.append([i, bin_list[2**x - 1 + y][1]])
                break

    if channel == 0:
        channel = 1
    else:
        channel = 0
print(len(bin_list))
