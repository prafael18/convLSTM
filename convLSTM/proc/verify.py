from scipy.misc import imread
from tensorflow.python.platform import gfile



filenames = gfile.Glob("out/epoch_1*")
cache=[]
for f in filenames:
  arr = imread(f)
  cache.append(arr)

frequency = [[] for x in cache]
filenames = gfile.Glob("out/*")
for f in filenames:
  arr = imread(f)
  res = [(arr==x).all() for x in cache]
  i = res.index(True)
  frequency[i].append(f)

print(frequency)
