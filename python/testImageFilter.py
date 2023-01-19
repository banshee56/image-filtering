import cv2
import numpy as np
import os
from scipy.ndimage import convolve
from myImageFilter import myImageFilter
from time import time

datadir    = '../data'      # the directory containing the images
resultsdir = '../results'   # the directory for dumping results

dummy_img = np.ndarray(shape=(3,10), buffer=np.array([90]*1000), dtype=float, order='F')
filter = np.ndarray(shape=(3,7), buffer=np.array([1/2]*100), dtype=float, order='F')

def readImg(file):
    file = os.path.splitext(file)[0]
    # read in images
    img = cv2.imread('%s/%s.jpg' % (datadir, file))
    
    if (img.ndim == 3):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    img = np.float32(img) / 255
    return img
    

# res = myImageFilter(dummy_img, filter)
# control = convolve(dummy_img, filter, mode='constant')
# print(res==control)
for file in os.listdir(datadir):
    if file.endswith('.jpg'):
        print(file)
        img = readImg(file)
        
        before = time()
        img1 = myImageFilter(img, filter)
        after = time()
        print(after-before)
        control = convolve(img, filter, mode='constant')

        # test against the convolution done by a library
        print(control)
        print(img1)
        print('-----------')
        if file=='img10.jpg':
            print(control==img1)