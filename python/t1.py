import numpy as np

# s = []
# for i in range(3):
#     a = np.random.randint(20, size=(3, 4))
#     s.append(a)

# s = np.array(s)
# print(s)

# print('----------')
# c = np.choose([2, 0, 1, 0], s)
# print(c)

# a = np.random.randint(1, 5, size=(5, 5))
# print(a)

# b = np.random.randint(1, 5, size=(5, 5))
# print(b)

# c = np.absolute(b-a)
# print(c)

# # non zero is because it was not the largest, so turn to 0
# li = np.nonzero(c)

# a[li] = 0
# s = np.stack([a, b, c])

# print('---------------------')
# g = np.random.randint(0, 3, size=(5, 5))
# print(g)
# f = np.choose(g, s)
# print(f)

def angleToIndex(g, m):
    # m should be 22.5
    n = g.copy()

    n = np.where((g < 135 + m) & (g >= m), n, 0)        # change angles in range [0, 22.5) and [157.5, 180] to 0 
    n = np.where((g < m) | (g >= 45 + m), n, 1)         # change angles in range [22.5, 67.5) to 1
    n = np.where((g < 90 - m) | (g >= 90 + m), n, 2)    # change angles in range [67.5, 112.5) to 2
    n = np.where((g < 135 - m) | (g >= 135 + m), n, 3)  # change angles in range [112.5, 157.5) to 3
  
    # FOR TESTING ONLY: checking if any angles not indexed
    # print(np.all((n == 0) | (n == 1) | (n == 2) | (n == 3)))
    
    return n

a = np.random.randint(0, 181, size=(3, 3))
print(a)

edge = np.array([0, 22.5, 157.5, 180, 67.5, 112.5, 157.5])
print(angleToIndex(a, 22.5))
