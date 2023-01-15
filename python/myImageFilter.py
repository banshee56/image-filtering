import numpy as np

def myImageFilter(img0, h):
    # how much to pad each axis by
    padRLength = h.shape[0]//2      # pad by half the number of rows in h
    padCLength = h.shape[1]//2      # pad by half the number of columns in h
    
    # padding the image with 0's
    padded = np.pad(img0, ((padRLength, padRLength), (padCLength, padCLength)), mode='constant')
    new = []                        # the values of the filtered image

    # go though the top left indices of each submatrix
    for i in range(img0.shape[0]):
        for j in range(img0.shape[1]):
            # compute submatrix where filter and padded image overlap
            submatrix = padded[i: h.shape[0]+i, j: h.shape[1]+j]

            # the value to replace pixel intensity with
            mul = np.multiply(submatrix, h)
            val = np.sum(mul)

            new.append(val)

    # turn the filtered image values into an ndarray (same dtype as orig image input)
    img1 = np.ndarray(shape=img0.shape, buffer=np.array(new))
    return img1

