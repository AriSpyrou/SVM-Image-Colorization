import glob

import cv2
import numpy as np

# Use multiple threads where possible
cv2.setUseOptimized(True)
cv2.setNumThreads(cv2.getNumberOfCPUs())


def discrete_colors(path='images/*.jpg'):
    filelist = glob.glob(path)
    imgs = []
    for path in filelist:
        tmp = cv2.imread(path)
        if tmp.shape[1] >= 400:
            scale = tmp.shape[1] // 300
        else:
            scale = 1
        tmp = cv2.resize(tmp, (int(tmp.shape[1] / scale), int(tmp.shape[0] / scale)))
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2Lab)
        imgs.append(tmp)
    del tmp, filelist

    ab = []
    for im in imgs:
        (l, a, b) = cv2.split(im)
        # Make a list with a, b coordinates

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                ab.append([a[i, j], b[i, j]])

    del l, a, b
    ab = np.asarray(ab)
    # median = np.median(ab, axis=0, overwrite_input=True)
    # ab = ab[ab[:, 0].argsort()]
    # Points at which the pixels are divided
    bins = [[0, ab.shape[0] - 1]]
    # Channel is 0 for alpha and 1 for beta
    channel = 0

    for x in range(4):
        for y in range(2 ** x):
            x_idx = 2 ** x - 1 + y
            median = np.median(ab[bins[x_idx][0]:bins[x_idx][1]], axis=0)
            ab[bins[x_idx][0]:bins[x_idx][1]] = ab[np.add(ab[bins[x_idx][0]:bins[x_idx][1], channel].argsort(),
                                                          np.full(bins[x_idx][1] - bins[x_idx][0], bins[x_idx][0]))]
            for i in range(bins[x_idx][0], bins[x_idx][1]):
                if ab[i][channel] >= median[channel]:
                    bins.append([bins[x_idx][0], i])
                    bins.append([i, bins[x_idx][1]])
                    break

        if channel == 0:
            channel = 1
        else:
            channel = 0

    for i in range(2 ** 4 - 1):
        bins.pop(0)

    colors = []
    for rn in bins:
        colors.append(np.median(ab[rn[0]:rn[1]], axis=0))

    return colors
