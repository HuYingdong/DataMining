import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

# Series
s = Series([1, 3, 6, np.nan, 44, 1])
s
s.values
s.index

obj = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj
obj.index
obj['d'] = 6
obj[['c', 'a', 'd']]
'b' in obj

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
obj3

states = ['california', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()
obj3 + obj4

obj4.name = 'population'
obj4.index.name = 'state'
obj4
obj.index = ['bob', 'steve', 'jeff', 'ryan']
obj

# DataFrame
df1 = DataFrame({'a': 1.,
                    'b': pd.Timestamp('20171125'),
                    'c': Series(1, index=list(range(4)), dtype='float32'),
                    'd': np.array([3] * 4, dtype='int32'),
                    'e': pd.Categorical(['test', 'train', 'test', 'train']),
                    'f': 'foo'},
                   columns=['f', 'e', 'd', 'c', 'b', 'a'])
df1
df1.dtypes
df1.index
df1.columns
df1.values
df1.describe()
df1.T
df1.sort_index(axis=1, ascending=False)
df1.sort_values(by='e')
df1['a']
df1.a
df1.ix[1]
df1['g'] = df1.b == '20171125'
del df1['g']

# reindex
obj = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)

obj2 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])

# 插值 ffill\bfill
obj2.reindex(range(6), method='ffill')

frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'],
                     columns=['ohio', 'texas', 'california'])
frame
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2
states = ['texas', 'utah', 'california']
frame.reindex(columns=states)
frame.reindex(index=['a', 'b', 'c', 'd'], columns=states)
frame.ix[['a', 'b', 'c', 'd'], states]  # 标签索引

# 删除 drop
obj = Series(np.arange(5.), index=[i for i in 'abcde'])
new_obj = obj.drop('c')
new_obj
obj.drop(['d', 'c'])

data = DataFrame(np.arange(16).reshape((4, 4)),
                    index=['ohio', 'colorado', 'utah', 'new tork'],
                    columns=['one', 'two', 'three', 'four'])
data
data.drop(['colorado', 'ohio'])
data.drop('two', axis=1)
data.drop(['two', 'four'], axis=1)

# 索引、选取和过滤
obj = Series(np.arange(4.), index=[i for i in 'abcd'])
obj['b']
obj[1]
obj[2:4]
obj[['b', 'a', 'd']]
obj[[1, 3]]
obj[obj < 2]
obj['b':'c']  # 末端包含（inclusive）
obj['b':'c'] = 5

df2 = DataFrame(np.arange(24).reshape((6, 4)),
                   index=pd.date_range('20170101', periods=6),
                   columns=(['a', 'b', 'c', 'd']))
df2['a']
df2.a
df2[['a', 'c']]
df2[:3]
df2['20170102':'2017-01-05']

# by label
df2.loc['20170103']
df2.loc[:, ['a', 'b']]
df2.loc['20170104':, ['a', 'b']]

# by position
df2.iloc[3:5, 1:3]

# mixed:ix
df2.ix[:3, ['a', 'c']]

# boolean indexing
df2[df2.a > 8]

df2.iloc[2, 2] = 0
df2.a[df2.a > 4] = 0
df2.b[df2.a > 4] = 0
df2['f'] = np.nan
df2['g'] = Series([1, 2, 3, 4, 5, 6],
                     index=pd.date_range('20170101', periods=6))

# ix
df2.ix[2]  # 行
df2.ix[:'2017-01-03', ['a', 'c']]
df2.ix[df2.a >= 12, :2]

# 算术运算和数据对齐
s1 = Series([7.3, -2.5, 3.4, 1.5], index=list('acde'))
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=[i for i in 'acefg'])
s1 + s2

df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                   index=['ohio', 'texas', 'colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                   index=['utah', 'ohio', 'texas', 'oregon'])
df1 + df2
# 算术方法 add sub div mul
df1.add(df2, fill_value=0)
df1.reindex(columns=df2.columns, fill_value=0)

# 广播 broadcasting
arr = np.arange(12.).reshape((3, 4))
arr - arr[0]

frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                     index=['utah', 'ohio', 'texas', 'oregon'])
series = frame.ix[0]
frame
series
frame - series

series2 = Series(range(3), index=list('bef'))
frame + series2
# 匹配行且在列上传播
series3 = frame['d']
frame.sub(series3, axis=0)

np.abs(frame)
# 应用函数到各列或各行 apply
f = lambda x: x.max() - x.min()
frame.apply(f)
frame.apply(f, axis=1)


def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])


frame.apply(f)

fmat = lambda x: '{0:.2f}'.format(x)
frame.applymap(fmat)
frame['e'].map(fmat)

# 排序与排名 sorting ranking
obj = Series(range(4), index=list('dabc'))
obj.sort_index()

frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'],
                     columns=list('dabc'))
frame.sort_index()
frame.sort_index(axis=1)
frame.sort_index(axis=1, ascending=False)

frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame.sort_index(by='b')
frame.sort_index(by=['a', 'b'])

obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()
obj.rank(method='first')
obj.rank(ascending=False, method='max')

frame.rank(axis=1)

# 带有重复值的轴索引
obj = Series(range(5), index=list('aabbc'))
obj
obj.index.is_unique
obj['a']

# 描述统计
df = DataFrame([[1.4, np.nan], [7.1, -4.5],
                   [np.nan, np.nan], [0.75, -1.3]],
                  index=['a', 'b', 'c', 'd'],
                  columns=['one', 'two'])
df
df.sum()
df.sum(axis=1)
df.mean(axis=1, skipna=False)
df.idxmax()
df.describe()
# corr cov corrwith

# 唯一值、值计数，成员资格
obj = Series(list('cadaabbcc'))
uniques = obj.unique()
uniques
uniques.sort()
obj.value_counts()
pd.value_counts(obj.values, sort=False)
mask = obj.isin(['b', 'c'])
mask
obj[mask]

# 处理丢失数据

dates = pd.date_range('20170101', periods=6)
df = DataFrame(np.arange(24).reshape((6, 4)),
                  index=dates,
                  columns=['a', 'b', 'c', 'd'])
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan

print(df.dropna(axis=0, how='any'))  # axis=1, how={'any', 'all'}
print(df.fillna(0))
df.fillna({1: 0.5, 2: -1})
print(df.isnull())
print(np.any(df.isnull()))

# 层次化索引
# 重排分级顺序
# 分级别汇总同统计

# 导入导出
data = pd.read_csv('\Python\Machine Learning\dates.csv')
df.to_csv('\Python\Machine Learning\dates.csv')
# read_excel read_hdf read_sql read_josn read_html read_pickle

# 合并 concat merge
df1 = DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
df3 = DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])

res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)  # 0竖向 1横向

# join ['inner', 'outer']
df1 = DataFrame(np.ones((3, 4)) * 0,
                   columns=['a', 'b', 'c', 'd'],
                   index=[1, 2, 3])
df2 = DataFrame(np.ones((3, 4)) * 1,
                   columns=['b', 'c', 'd', 'e'],
                   index=[2, 3, 4])
res = pd.concat([df1, df2], join='outer')
res = pd.concat([df1, df2], join='inner', ignore_index=True)

# join_axes
res = pd.concat([df1, df2], axis=1)
res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])

# append
df1 = DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
res = df1.append(df2, ignore_index=True)
res = df1.append([df2, df2], ignore_index=True)
S1 = Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
res = df1.append(S1, ignore_index=True)

# merging two df by key
left = DataFrame({'key': ['k0', 'k1', 'k2', 'k3'],
                     'a': ['a0', 'a1', 'a2', 'a3'],
                     'b': ['b0', 'b1', 'b2', 'b3']})
right = DataFrame({'key': ['k0', 'k1', 'k2', 'k3'],
                      'c': ['c0', 'c1', 'c2', 'c3'],
                      'd': ['d0', 'd1', 'd2', 'd3']})

res = pd.merge(left, right, on='key')

# consider two keys
left = DataFrame({'key1': ['k0', 'k0', 'k1', 'k2'],
                     'key2': ['k0', 'k1', 'k0', 'k1'],
                     'a': ['a0', 'a1', 'a2', 'a3'],
                     'b': ['b0', 'b1', 'b2', 'b3']})
right = DataFrame({'key1': ['k0', 'k1', 'k1', 'k2'],
                      'key2': ['k0', 'k0', 'k0', 'k0'],
                      'c': ['c0', 'c1', 'c2', 'c3'],
                      'd': ['d0', 'd1', 'd2', 'd3']})
# how= ['left', 'right', 'outer', 'inner'(default)]
res = pd.merge(left, right, on=['key1', 'key2'], how='inner')
res = pd.merge(left, right, on=['key1', 'key2'], how='outer')
res = pd.merge(left, right, on=['key1', 'key2'], how='right')

# indicator
df1 = DataFrame({'col1': [0, 1], 'col_left': ['a', 'b']})
df2 = DataFrame({'col1': [1, 2, 2], 'col_right': [2, 2, 2]})

res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
res = pd.merge(df1, df2, on='col1', how='outer',
               indicator='indicator_column')

# index
left = DataFrame({'a': ['a0', 'a1', 'a2', 'a3'],
                     'b': ['b0', 'b1', 'b2', 'b3']},
                    index=['k0', 'k1', 'k2', 'k3'])
right = DataFrame({'c': ['c0', 'c1', 'c2', 'c3'],
                      'd': ['d0', 'd1', 'd2', 'd3']},
                     index=['k0', 'k2', 'k3', 'k4'])

res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
res = pd.merge(left, right, left_index=True, right_index=True, how='inner')

# handle overlapping
boys = DataFrame({'k': ['k0', 'k1', 'k2'], 'age': [1, 2, 3]})
girls = DataFrame({'k': ['k0', 'k0', 'k3'], 'age': [4, 5, 6]})

res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='outer')

# pandas plot

# Series
data = Series(np.random.randn(1000), index=np.arange(1000))
data = data.cumsum()
data.plot()
plt.show()

# DataFrame
data = DataFrame(np.random.randn(1000, 4),
                    index=np.arange(1000),
                    columns=list('abcd'))

data = data.cumsum()
print(data.head())
data.plot()
plt.show()

# plot methods: bar, hist, box, kde, area, scatter, hexbin, pie
ax = data.plot.scatter(x='a', y='b', color='DarkBlue', label='Class 1')
data.plot.scatter(x='a', y='d', color='DarkGreen', label='Class 2', ax=ax)
plt.show()
