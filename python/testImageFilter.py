import cv2
import numpy as np
import os
from scipy.ndimage import convolve
from myImageFilter import myImageFilter


datadir    = '../data'      # the directory containing the images
resultsdir = '../results'   # the directory for dumping results

dummy_img = np.ndarray(shape=(5,5), buffer=np.array([90]*49), dtype=int, order='F')
filter = np.ndarray(shape=(3,3), buffer=np.array([1/2]*49), dtype=float, order='F')

def readImg(file):
    file = os.path.splitext(file)[0]
    # read in images
    img = cv2.imread('%s/%s.jpg' % (datadir, file))
    
    if (img.ndim == 3):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    img = np.float32(img) / 255
    return img
    

# myImageFilter(dummy_img, filter)
for file in os.listdir(datadir):
    if file.endswith('.jpg'):
        print(file)
        img = readImg(file)
        img1 = myImageFilter(img, filter)
        control = convolve(img, filter, mode='constant')

        # test against the convolution done by a library
        print(control)
        print(img1)
        print('-----------')
        if file=='img10.jpg':
            print(control==img1)