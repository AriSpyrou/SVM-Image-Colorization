import cv2
from skimage.segmentation import slic

im = cv2.imread('castle.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
segments = slic(im, n_segments=50, convert2lab=True)
surf = cv2.SURF(400)
des = cv2.xfeatures2d_SURF.detect()
print('ok')