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

elem0 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])        
elem45 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
elem90 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
elem135 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print(elem0)
print(elem45)
print(elem90)
print(elem135)

