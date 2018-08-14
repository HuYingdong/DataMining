import random
import numpy as np

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
arr2d[:2]
arr2d[:2, 1:]

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
from numpy.linalg import inv, qr, eig

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