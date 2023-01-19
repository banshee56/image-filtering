import numpy as np

a = np.array([[1, 2, 9], [4, 8, 5], [9, 6, 7]])
print(a)
b = a.flatten()

v = np.argpartition(b, -8)[-8:]
print(v)

print(a)
for i in v:
    x = int(i / 3)
    y = i % 3

    print(a[x, y])



# print(a)
# top = np.argpartition(a, 1)[]
# print(top)



# def k_largest_index_argpartition_v1(a, k):
#     idx = np.argpartition(-a.ravel(),k)[:k]
#     return np.column_stack(np.unravel_index(idx, a.shape))

# for i in k_largest_index_argpartition_v1(a, 3):
#     print(a[i[0], i[1]])