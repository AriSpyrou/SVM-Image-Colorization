import glob

import cv2
import numpy as np
from joblib import dump, load
from skimage.segmentation import slic
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import colour_space_discretization as csd
import gabor as gb

LOAD = True

if not LOAD:
    colors = csd.discrete_colors('images/*.jpg')
    filelist = glob.glob('images/*.jpg')
    img = []
    for path in filelist:
        tmp = cv2.imread(path)

        if tmp.shape[1] >= 400:
            scale = tmp.shape[1] // 300
        else:
            scale = 1
        tmp = cv2.resize(tmp, (int(tmp.shape[1] / scale), int(tmp.shape[0] / scale)))

        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2Lab)

        h, w, d = tuple(tmp.shape)
        tmp_array = np.reshape(tmp, (w * h, d))
        tmp_array = np.delete(tmp_array, 0, 1)
        labels = pairwise_distances_argmin(colors, tmp_array, axis=0)
        label_idx = 0
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                tmp[i, j, 1:] = colors[labels[label_idx]]
                label_idx += 1
        '''
        sanity = cv2.cvtColor(tmp_img, cv2.COLOR_Lab2BGR)
        cv2.imshow('i', sanity)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('test.png', sanity)
        '''
        img.append(tmp)
    del tmp, filelist

    filters = gb.build_filters()
    # Init SURF object
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, extended=False, upright=True)
    X = np.array([])
    y = []
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
            tmp = []
            # For each point in segment
            for pt in seg:
                # Add keypoint to temp list of kps
                kp.append(cv2.KeyPoint(pt[1], pt[0], 1))
                tmp.append([im[pt[0], pt[1], 1], im[pt[0], pt[1], 2]])
            # Extract median colour from each superpixel
            y.append(np.median(tmp, axis=0))
            # Compute feature descriptors and put into temp array
            des_tmp = surf.compute(im, kp, None)
            # Find mean of feature descriptors effectively computing
            # a feature descriptor for each superpixel
            surf_feat[f] = np.mean(des_tmp[1], axis=0)
            f += 1

        gabor_feat = np.zeros((2, len(grp_segs), filters.shape[0]))
        f = 0
        for filter in filters:
            resp = gb.process(im[:, :, 0], filter)
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
        tmp = np.block([surf_feat, gabor_feat[0], gabor_feat[1]])
        if X.size == 0:
            X = tmp
        else:
            X = np.vstack((X, tmp))

    y = pairwise_distances_argmin(colors, np.asarray(y), axis=0)
    sc = StandardScaler()
    pca = PCA()
    X = pca.fit_transform(sc.fit_transform(X))
    lin_clf = LinearSVC(dual=False, fit_intercept=False)
    lin_clf.fit(X, y)
    dump(lin_clf, 'linear_classifier.joblib')
else:
    lin_clf = load('linear_classifier.joblib')
