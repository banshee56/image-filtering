import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):
    rhoScale = []   # an array with the lower boundary of each interval of rhos (i.e. [0, 5, 10, 15, ...] would result from
    # rhoRes=5, then the first index corresponds to the bucket with rho values [0, 4])
    thetaScale = []
    
    # the maximum possible value for rho (distance) is the diagonal of the image
    diagonal = np.hypot(Im.shape[0], Im.shape[1])
    rhoScale = list(np.arange(0, diagonal, rhoRes))
    thetaScale = list(np.arange(0, 2*np.pi, thetaRes))      # from 0 to 2pi as mentioned in the instructions
    img_hough = np.zeros(shape=(len(rhoScale), len(thetaScale)))

    for x in range(Im.shape[1]):
        for y in range(Im.shape[0]):
            # an edge pixel
            if Im[y, x] != 0:
                for theta in thetaScale:
                    rho = x * np.cos(theta) + y * np.sin(theta)

                    # find the index of the bucket with the largest value smaller than rho
                    if rho >= 0:
                        rhoIndex = np.searchsorted(rhoScale, rho, 'left')
                        thetaIndex = thetaScale.index(theta)
                        img_hough[rhoIndex][thetaIndex] += 1
    
    return img_hough, rhoScale, thetaScale
