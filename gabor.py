import cv2
import numpy as np


def build_filters(ksize=(9, 9), orientations=16):
    filters = []
    for theta in np.arange(0, orientations * np.pi / orientations, np.pi / orientations):
        kern = cv2.getGaborKernel(ksize=ksize, sigma=1.5, theta=theta, lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return np.asarray(filters)


def process(im, filters):
    accum = np.zeros_like(im)
    for kern in filters:
        fimg = cv2.filter2D(im, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


'''
if __name__ == '__main__':
    # Create filters with default settings
    # and convert to ndarray
    filters = build_filters()
    # Read the image and init a feature vector
    im = cv2.imread('castle.jpg', 0)
    feat = []
'''
