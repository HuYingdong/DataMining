import random
import time
import numpy as np

a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
type(a)
arr = np.array(a)
type(arr)

a1 = [1, 3, 4]
array1 = np.array(a1, dtype=np.int32)
print(array1.dtype)
print(array1.ndim)
print(array1.shape)
print(array1.size)

b1 = [4, 5, 6]
a2 = [a1, b1]
array2 = np.array(a2)

b2 = [[7, 8, 9], [10, 11, 12]]
a3 = (a2, b2)
array3 = np.array(a3)

np.array([1, 2, 3], dtype=float)

np.array([1.1, 2.5, 3.3], dtype=int)

np.zeros(shape=(2, 5))
np.zeros_like(array3)
np.ones(shape=(3, 4))
np.ones_like(array3)
np.empty((3, 4))
np.empty_like(array3)

np.arange(start=10, stop=30, step=5)
np.arange(0, 2, 0.3)
np.linspace(start=0, stop=2, num=9)

# max()
x = np.array([[1, 2, 3], [4, 5, 6]])
x
x.flat
x.flatten()
x.max()
x.max(axis=0)
x.max(axis=1)
# argmax()
x.argmax()
x.argmax(axis=0)
x.argmax(axis=1)
# dot() 矩阵乘法
y = np.array([[1, 3], [2, 4], [5, 6]])
y
x.dot(y)
np.dot(x, y)
np.cumsum(y)
np.diff(y)

# ravel 向量
x.ravel()
x.ravel()[x.argmax()]
# transpose()
x.transpose()
x.T
# reshape()
x.reshape((1, 6))
x.reshape((6, 1))
x.reshape((6,))

# deep and shallow copies
x = np.array([[1, 2, 3], [4, 5, 6]])
y = x
x[0, 0] = 100
y
z = x.copy()
z
x[0, 0] = 1
z

# 对应元素运算
x = np.array([[1, 2, 3], [4, 5, 6]])
y = x
x + y
x - y
x * y
x / y
x + 1
x * 2
x ** 2
z = np.array([1, 2])
x + z.reshape((2, 1))

A = np.arange(1, 25).reshape((4, 6))
A = A * A

m = A.mean(axis=1).reshape(4, 1)
B = A - m
A2 = B * B
var1 = A2.mean(axis=1)
var1

x = np.array([[1, 2, 3], [4, 5, 6]])
np.log(x)
np.exp(x)

# linear algebra
# norm()
# inv()
# solve()
# det()
# eig()

from numpy.linalg import norm, inv, solve, det, eig
A = np.array([[1, 2], [3, 4]])
x = np.array([5, 6])
norm(A)
norm(x)
inv(A)
solve(A, x)
det(A)
v = eig(A)
v
v[0]
v[1]


def mydot(A, B):
    c = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            s = 0
            for k in range(A.shape[1]):
                s += A[i, k] * B[k, j]
                c[i, j] = s
    return c


A = np.random.randn(100, 100)
B = np.random.rand(100, 100)

last_time = time.time()
np.dot(A, B)
print(time.time() - last_time)

last_time = time.time()
mydot(A, B)
print(time.time() - last_time)

# merge

a = np.array([1, 1, 1])
b = np.array([2, 2, 2])
c = np.vstack((a, b))
c.shape
d = np.hstack((a, b))
d.shape

a[np.newaxis, :].shape
a[:, np.newaxis].shape
e = np.concatenate((a, b, a, b), axis=0)
A = np.arange(1, 25).reshape((4, 6))
np.split(A, 2, axis=0)
np.array_split(A, 4, axis=1)
np.vsplit(A, 2)
np.hsplit(A, 3)

# 数组数据处理
points1 = np.arange(-5, 5, 1)
points2 = np.arange(0, 10, 1)
xs, ys = np.meshgrid(points1, points2)
xs.shape

import matplotlib.pyplot as plt
z = np.sqrt(xs ** 2 + ys ** 2)
plt.imshow(z, cmap=plt.cm.gray);
plt.colorbar()

arr = np.random.randn(4, 4)
arr
np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr)

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr.cumsum(1)
arr.cumprod(1)
# sum mean std var argmin argmax cumsum cumprod

# 布尔型数组
arr = np.random.randn(100)
(arr > 0).sum()
# any 是否存在一个或多个True
# all 数组所有值是否都是True

# sort
arr = np.random.rand(5, 3)
print(arr)
arr.sort(1)
print(arr)

id = np.array([1, 2, 3, 1, 2, 3])
np.unique(id)

# 导入导出
np.save('testarray', arr)
np.load('testarray.npy')
np.savez('test_archive.npz', a=arr, b=arr)
arch = np.load('test_archive.npz')
print(arch['a'])
np.savetxt('test_array.txt', arr, delimiter=',')


# 索引
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]
arr2d[0][2]
arr2d[0, 2]

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d
arr3d[0]
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d
arr3d[1, 0]

# 切片索引
A = array2
A
A[0, 0]
A[0, 2]
A[1, -1]
A[1, -2]

A = np.arange(1, 81).reshape(10, 8)
A
A[2:8, 3:7]
A[:, 4:]
A[:4, :5]
A[7:, 4:]

C = np.arange(1, 21)
C
C[:15]
C[3:15]
C[:15:2]
C[10::3]
C[::3]

A[:, 1::2]
A[:7:2, :]

A = np.arange(1, 101).reshape(10, 10)
A
A11 = A[:5, :5]
A11
A12 = A[:5, 5:]
A12
A21 = A[5:, :5]
A21
A22 = A[5:, 5:]
A22

B1 = A[::2, :]
B1
B2 = A[:, 1::2]
B2

# 花式索引
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr[[4, 3, 0, 6]]
arr[[-3, -5, -7]]

arr = np.arange(32).reshape((8, 4))
# （1, 0） (5, 3) (7, 1) (2, 2)
arr[[1, 5, 7, 2], [0, 3, 1, 2]]

arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]
arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]

# np.where 条件逻辑
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
result = np.where(cond, xarr, yarr)
arr = np.random.randn(4, 4)
arr
np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr)  # 只将正值设为2

# 数学和统计方法
arr = np.random.randn(5, 4)
arr.mean()
arr.mean(axis=0)
arr.sum(axis=1)
arr.std()
arr.cumsum(axis=0)
arr.cumprod(axis=0)
arr.argmax(axis=0)

# 排序
arr = np.random.rand(8)
arr.sort()
arr = np.random.randn(5, 3)
arr.sort(axis=0)

large_arr = np.random.rand(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]  # 5%分位数

# 集合
ints = np.array([3, 3, 3, 2, 2, 2, 1, 1, 4, 4])
np.unique(ints)

values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])

# 读写数据
arr = np.arange(10)
np.save('./test_arr', arr)
np.load('./test_arr.npy')

arr = np.arange(24).reshape((6, 4))
np.savetxt('test_arr.txt', arr, delimiter=',')
np.loadtxt('test_arr.txt', delimiter=',')

# 线性代数
from numpy.linalg import inv, qr, eig, norm

x = np.random.rand(5, 5)
mat = x.T.dot(x)
inv(mat)
mat.dot(inv(mat))
q, r = qr(mat)
eig_value, eig_vector = eig(mat)

# 随机数
samples = np.random.normal(loc=10, scale=100, size=(4, 4))

# 随机游走
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

draws = np.random.randint(0, 2, size=1000)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
walk.min()
walk.max()
(np.abs(walk) >= 10).argmax()

# 多个随机游走
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
# steps = np.random.normal(loc=0, scale=0.25, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks.max()
walks.min()

hits30 = (np.abs(walks) >= 30).any(1)
hits30.sum()

crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times.mean()