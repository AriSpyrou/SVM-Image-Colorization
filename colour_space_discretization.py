import glob

import cv2
import numpy as np

# Use multiple threads where possible
cv2.setUseOptimized(True)
cv2.setNumThreads(cv2.getNumberOfCPUs())


def discrete_colors(path='images/train/*.jpg'):
    filelist = glob.glob(path)
    imgs = []
    # Import all images from path and resize them to be of manageable size
    for path in filelist:
        tmp = cv2.imread(path)
        if tmp.shape[1] >= 400:
            scale = tmp.shape[1] // 300
        else:
            scale = 1
        tmp = cv2.resize(tmp, (int(tmp.shape[1] / scale), int(tmp.shape[0] / scale)))
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2Lab)
        imgs.append(tmp)
    del filelist

    ab = []
    for im in imgs:
        (l, a, b) = cv2.split(im)
        # Fill ab with alpha and beta pairs

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                ab.append([a[i, j], b[i, j]])

    del l, a, b
    # Convert ab list to numpy array
    ab = np.asarray(ab)

    # List which represents each bin
    bins = [[0, ab.shape[0] - 1]]

    # Channel is 0 for alpha and 1 for beta
    channel = 0

    for x in range(4):
        for y in range(2 ** x):
            # Index in bins
            x_idx = 2 ** x - 1 + y
            # Find median
            median = np.median(ab[bins[x_idx][0]:bins[x_idx][1]], axis=0)
            # Partially sort the numpy array
            ab[bins[x_idx][0]:bins[x_idx][1]] = ab[np.add(ab[bins[x_idx][0]:bins[x_idx][1], channel].argsort(),
                                                          np.full(bins[x_idx][1] - bins[x_idx][0], bins[x_idx][0]))]
            # Split the bin into two bins
            for i in range(bins[x_idx][0], bins[x_idx][1]):
                if ab[i][channel] >= median[channel]:
                    bins.append([bins[x_idx][0], i])
                    bins.append([i, bins[x_idx][1]])
                    break
        # Switch channel for next iteration
        if channel == 0:
            channel = 1
        else:
            channel = 0
    # Remove the first 15 bins
    for i in range(2 ** 4 - 1):
        bins.pop(0)

    # Init a list of colours, find the median value of each bin, append it to
    # the list and return it
    colors = []
    for rn in bins:
        colors.append(np.median(ab[rn[0]:rn[1]], axis=0))

    return colors
