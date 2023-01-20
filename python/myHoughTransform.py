import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):
    diagonal = np.hypot(Im.shape[0], Im.shape[1])                   # the maximum possible value for rho (distance) is the diagonal of the image
    rhoScale = np.arange(0, diagonal, rhoRes)           # starting from the average of rhoRes to get the average values of the buckets
    thetaScale = np.arange(0, 2*np.pi, thetaRes)                    # from 0 to 2pi as mentioned in the instructions
    img_hough = np.zeros(shape=(len(rhoScale), len(thetaScale)))
    edgePoints = np.transpose(np.nonzero(Im))

    for point in edgePoints:
        x = point[1]
        y = point[0]
        
        rhos = x * np.cos(thetaScale) + y * np.sin(thetaScale)
        thetaInd = np.where(rhos >= 0)       # get indices of positive rhos, these correspond to theta indices that led to positive rhos
        rhos = rhos[rhos >= 0]               # keep only the positive rho values in the array
        rhoInd = np.digitize(rhos, rhoScale)-1
        img_hough[rhoInd, thetaInd] += 1
    
    return img_hough, rhoScale, thetaScale

# def myHoughTransform(Im, rhoRes, thetaRes):
#     rhoScale = np.array([])   # an array with the UPPER boundary of each interval of rhos (i.e. [5, 10, 15, 20, ...] would result from
#     # rhoRes=5, with the first index corresponding to the bucket with rho values [0, 5))
#     thetaScale = np.array([])
    
#     # the maximum possible value for rho (distance) is the diagonal of the image
#     diagonal = np.hypot(Im.shape[0], Im.shape[1])
#     rhoScale = list(np.arange(0 + rhoRes, diagonal + rhoRes, rhoRes))
#     thetaScale = list(np.arange(0, 2*np.pi, thetaRes))      # from 0 to 2pi as mentioned in the instructions
#     img_hough = np.zeros(shape=(len(rhoScale), len(thetaScale)))
#     n = np.count_nonzero(Im)
#     print(n)
#     for x in range(Im.shape[1]):
#         for y in range(Im.shape[0]):
#             # an edge pixel has value 1 on img_threshold
#             if Im[y, x] != 0:
#                 # go through all theta values
#                 for thetaIndex in range(len(thetaScale)):
#                     theta = thetaScale[thetaIndex]
#                     rho = x * np.cos(theta) + y * np.sin(theta)

#                     # ignore theta values corresponding to negative rhos
#                     if rho >= 0:
#                         # find the index of the bucket with the largest value *smaller* than rho
#                         rhoIndex = np.searchsorted(rhoScale, rho, 'left')
#                         img_hough[rhoIndex][thetaIndex] += 1
    
#     return img_hough, rhoScale, thetaScale
