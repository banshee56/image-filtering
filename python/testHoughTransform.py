import cv2
import numpy as np
import os

from myEdgeFilter import myEdgeFilter
from myHoughLines import myHoughLines
from myHoughTransform import myHoughTransform

datadir    = '../data'      # the directory containing the images
resultsdir = '../results'   # the directory for dumping results

# parameters
sigma     = 2
threshold = 0.03
rhoRes    = 2
thetaRes  = np.pi / 90
nLines    = 15
# end of parameters


# read in images
file = 'img01'
img = cv2.imread('%s/%s.jpg' % (datadir, file))

if (img.ndim == 3):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img = np.float32(img) / 255

# actual Hough line code function calls
img_edge = myEdgeFilter(img, sigma)
img_threshold = np.float32(img_edge > threshold)
[img_hough, rhoScale, thetaScale] = myHoughTransform(img_threshold, rhoRes, thetaRes)
# everything below here just saves the outputs to files        
fname = '%s/%s_03hough.png' % (resultsdir, file)
cv2.imwrite(fname, 255 * img_hough / img_hough.max())
