import copy
import glob

import cv2
import numpy as np
from skimage.segmentation import slic
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import colour_space_discretization as csd
import gabor as gb

N_SEG = 50
SLIC_COMP = 10


def colorize(img, colors):
    # Create the segments using SLIC
    segments = slic(img, n_segments=N_SEG, compactness=SLIC_COMP, convert2lab=True)

    # Fill lists with the coords of the pixels of each segment
    grp_segs = [[] for k in range(segments.max() + 1)]
    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            grp_segs[segments[i, j]].append([i, j])
    sp_idx = 0
    for seg in grp_segs:
        for pt in seg:
            img[pt[0], pt[1], 1:] = colormap[colors[sp_idx]]
        sp_idx += 1
    return img


def process_image(path, convert2lab=True, resize=True, quantize=True):
    # Load temp image
    tmp_img = cv2.imread(path)
    im = tmp_img
    if resize:
        # Resize if too big
        if tmp_img.shape[1] >= 400:
            scale = tmp_img.shape[1] // 300
        else:
            scale = 1
        tmp_img = cv2.resize(tmp_img, (int(tmp_img.shape[1] / scale), int(tmp_img.shape[0] / scale)))

    if convert2lab:
        # Convert to Lab
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2Lab)

    if quantize:
        # Reshape image to 2D array and delete the L channel
        h, w, d = tuple(tmp_img.shape)
        tmp_array = np.reshape(tmp_img, (w * h, d))
        tmp_array = np.delete(tmp_array, 0, 1)
        # Find the most similar colour in the colormap and replace it on the image
        labels = pairwise_distances_argmin(colormap, tmp_array, axis=0)
        label_idx = 0
        for i in range(tmp_img.shape[0]):
            for j in range(tmp_img.shape[1]):
                tmp_img[i, j, 1:] = colormap[labels[label_idx]]
                label_idx += 1
    return tmp_img


def features_and_labels(images):
    X = np.array([])
    y = []

    for img in images:
        # Create the segments using SLIC
        segments = slic(img, n_segments=N_SEG, compactness=SLIC_COMP, convert2lab=True)

        # Fill lists with the coords of the pixels of each segment
        grp_segs = [[] for k in range(segments.max() + 1)]
        for i in range(segments.shape[0]):
            for j in range(segments.shape[1]):
                grp_segs[segments[i, j]].append([i, j])

        '''  
        For each superpixel add its pixels as keypoints and compute surf features for
        each keypoint afterwards compute the surf descriptor of each superpixel by 
        averaging all the descriptors of its pixels
        '''
        # Init surf feature vector
        surf_feat = np.zeros((len(grp_segs), 128))

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
                tmp.append([img[pt[0], pt[1], 1], img[pt[0], pt[1], 2]])
            # Extract median colour from each superpixel
            y.append(np.median(tmp, axis=0))
            # Compute feature descriptors and put into temp array
            des_tmp = surf.compute(img[:, :, 0], kp, None)
            # Find mean of feature descriptors effectively computing
            # a feature descriptor for each superpixel
            surf_feat[f] = np.mean(des_tmp[1], axis=0)
            f += 1

        gabor_feat = np.zeros((2, len(grp_segs), filters.shape[0]))
        f = 0
        for filter in filters:
            resp = gb.process(img[:, :, 0], filter)
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
    return X, y


if __name__ == '__main__':
    # Create the colormap from images in the directory
    colormap = csd.discrete_colors('images/train/*.jpg')
    # Init SURF object
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, extended=True)
    # Create the Gabor filters
    filters = gb.build_filters()
    filelist = glob.glob('images/train/*.jpg')
    images = []

    for path in filelist:
        images.append(process_image(path))
    del filelist

    # Create X, y testing data
    X_train, y_train = features_and_labels(images)

    # Create test data
    filelist = glob.glob('images/test/*.jpg')
    images_test = []
    for path in filelist:
        images_test.append(process_image(path))
    X_test, y_test = features_and_labels(images_test)

    y_train = pairwise_distances_argmin(colormap, np.asarray(y_train), axis=0)
    pca = PCA()
    sc = StandardScaler()
    X_train = pca.fit_transform(sc.fit_transform(X_train))
    X_test = pca.transform(sc.transform(X_test))

    lin_clf = LinearSVC(dual=False, fit_intercept=False)
    lin_clf.fit(X_train, y_train)

    colors = lin_clf.predict(X_test)
    original = copy.copy(images_test[0])
    colorize(images_test[0], colors)
    results = np.concatenate((original, images_test[0]), axis=1)
    results = cv2.cvtColor(results, cv2.COLOR_Lab2BGR)
    cv2.imshow('Original(left) and Colorized(right)', results)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
