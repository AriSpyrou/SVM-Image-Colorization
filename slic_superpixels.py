import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
from skimage.util import img_as_float

# im = cv2.imread('castle.jpg')
im = img_as_float(io.imread('castle.jpg'))
segments = slic(im, 100, compactness=50, sigma=2)
fig = plt.figure("SLIC Superpixels -- 100 segments")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(im, segments))
plt.axis("off")

plt.show()
