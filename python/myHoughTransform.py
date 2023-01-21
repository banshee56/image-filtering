import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):
    diagonal = np.hypot(Im.shape[0], Im.shape[1])           # the maximum possible value for rho (distance) is the diagonal of the image
    rhoScale = np.arange(0+rhoRes, diagonal+rhoRes, rhoRes) # arranging rhoScale to work with np.digitize(), i.e. using upper boundary of each bin
    thetaScale = np.arange(0, 2*np.pi, thetaRes)            # from 0 to 2pi as mentioned in the instructions

    # the accumulator
    img_hough = np.zeros(shape=(len(rhoScale), len(thetaScale)))
    edgePoints = np.transpose(np.nonzero(Im))   # the edge points are non-zero, transposed to make points easy to index with

    # go through each edge point
    for point in edgePoints:
        x = point[1]
        y = point[0]
        
        # calculate rhos corresponding to the thetas in thetaScale
        rhos = x * np.cos(thetaScale) + y * np.sin(thetaScale)
        thetaInd = np.where(rhos >= 0)          # get indices of positive rhos, these correspond to theta indices that produced positive rhos
        rhos = rhos[rhos >= 0]                  # keep only the positive rho values in the array
        rhoInd = np.digitize(rhos, rhoScale)    # getting bin indices
        img_hough[rhoInd, thetaInd] += 1        # accumulating votes
    
    return img_hough, rhoScale, thetaScale
