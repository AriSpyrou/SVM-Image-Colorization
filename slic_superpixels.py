import glob

import cv2
import numpy as np
from skimage.segmentation import slic
from sklearn import preprocessing

import gabor as gb

filelist = glob.glob('images/*.jpg')
img = []
for path in filelist:
    tmp = cv2.imread(path)
    if tmp.shape[1] >= 400:
        scale = tmp.shape[1] // 200
    else:
        scale = 1
    tmp = cv2.resize(tmp, (int(tmp.shape[0] / scale), int(tmp.shape[1] / scale)))
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2Lab)
    img.append(tmp)
del tmp, filelist

filters = gb.build_filters()
# Init SURF object
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, extended=False, upright=True)
for im in img:
    # Create the segments using SLIC
    segments = slic(im, n_segments=50, convert2lab=True)
    # Group each superpixel's individual pixel's coordinates in lists
    grp_segs = [[] for k in range(segments.max() + 1)]
    # Fill lists with the coords of the pixels of each segment
    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            grp_segs[segments[i, j]].append([i, j])

    '''
    
    For each superpixel add its pixels as keypoints and compute surf features for
    each keypoint afterwards compute the surf descriptor of each superpixel by 
    averaging all the descriptors of its pixels
    
    '''

    surf_feat = np.zeros((len(grp_segs), 64))
    # For each segment
    f = 0
    for seg in grp_segs:
        # Init temp keypoint list
        kp = []
        # For each point in segment
        for pt in seg:
            # Add keypoint to temp list of kps
            kp.append(cv2.KeyPoint(pt[1], pt[0], 1))
        # Compute feature descriptors and put into temp array
        des_tmp = surf.compute(im, kp, None)
        # Find mean of feature descriptors effectively computing
        # a feature descriptor for each superpixel
        surf_feat[f] = np.mean(des_tmp[1], axis=0)
        f += 1

    # Convert the image from Lab to BGR and then from BGR to Grayscale
    im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_Lab2BGR), cv2.COLOR_BGR2GRAY)
    gabor_feat = np.zeros((2, len(grp_segs), filters.shape[0]))
    f = 0
    for filter in filters:
        resp = gb.process(im, filter)
        p = 0
        for seg in grp_segs:
            tmp_le = 0
            tmp_ma = 0
            for pt in seg:
                tmp_le += resp[pt[0], pt[1]] ** 2
                tmp_ma += abs(resp[pt[0], pt[1]])
            gabor_feat[0, p, f] = tmp_le
            gabor_feat[1, p, f] = tmp_ma
            p += 1
        f += 1
    scaler = preprocessing.StandardScaler()
    scaled_surf = scaler.fit_transform(surf_feat)
    # TODO sotiro concat - pca - svm
