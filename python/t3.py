import numpy as np

a = np.array([3, 5, 6, 2, 1, 8, 0])
buckets = np.arange(0, 10, 2)   
# ind = np.searchsorted(buckets, a, 'right')
ind = np.digitize(a, buckets)-1

print(ind)