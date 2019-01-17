import cv2
import numpy as np
from skimage.segmentation import slic

# Read image and convert to Lab
im = cv2.imread('castle.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
# Create the segments using SLIC
segments = slic(im, n_segments=50, convert2lab=True)
# Init SURF class using parameters
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, extended=False, upright=True)
# Init lists for descriptors
des = []
kp = []

'''
# Iterate over the image and append each pixel as a keypoint
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        kp.append(cv2.KeyPoint(i, j, 1))
# Compute a feature descriptor for each keypoint and delete kp list from memory
des = surf.compute(im, kp, None)
del kp
'''
# Init list of lists
grp_segs = [[] for k in range(segments.max() + 1)]
# Fill lists with the coords of the pixels of each segment
for i in range(segments.shape[0]):
    for j in range(segments.shape[1]):
        grp_segs[segments[i, j]].append([i, j])

# For each segment
for seg in grp_segs:
    # Init temp keypoint list
    kp_tmp = []
    # For each point in segment
    for pt in seg:
        # Add keypoint to temp list of kps
        kp_tmp.append(cv2.KeyPoint(pt[1], pt[0], 1))
    # Compute feature descriptors and put into temp array
    des_tmp = surf.compute(im, kp_tmp, None)
    # Find mean of feature descriptors effectively computing
    # a feature descriptor for each superpixel
    des.append(np.mean(des_tmp[1], axis=0))
print('ok')
