import cv2
import numpy as np
import os

from myEdgeFilter import myEdgeFilter
from myHoughLines import myHoughLines
from myHoughTransform import myHoughTransform

datadir    = '../results'      # the directory containing the images
resultsdir = '../results'   # the directory for dumping results

# parameters
sigma     = 2
threshold = 0.03
rhoRes    = 2
thetaRes  = np.pi / 90
nLines    = 15
# end of parameters


# read in images
file = 'img03_03hough'
img_hough = cv2.imread('%s/%s.png' % (datadir, file))

[rhos, thetas] = myHoughLines(img_hough, nLines)

lines = cv2.HoughLinesP(np.uint8(255 * img_threshold), rhoRes, thetaRes, \
                        50, minLineLength = 20, maxLineGap = 5)

fname = '%s/%s_04lines.png' % (resultsdir, file)
img_lines = np.dstack([img,img,img])

# display line results from myHoughLines function in red
for k in np.arange(nLines):
    a = np.cos(thetaScale[thetas[k]])
    b = np.sin(thetaScale[thetas[k]])
    
    x0 = a*rhoScale[rhos[k]]
    y0 = b*rhoScale[rhos[k]]
    
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    
    cv2.line(img_lines,(x1,y1),(x2,y2),(0,0,255),1)

# display line segment results from cv2.HoughLinesP in green
for line in lines:
    coords = line[0]
    cv2.line(img_lines, (coords[0], coords[1]), (coords[2], coords[3]), \
                (0, 255, 0), 1)

cv2.imwrite(fname, 255 * img_lines)

