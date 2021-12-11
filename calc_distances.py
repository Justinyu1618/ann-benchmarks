from ann_benchmarks.algorithms.annoy import Annoy
from ann_benchmarks.distance import metrics
import h5py 
import numpy as np

prev_dataset = h5py.File("./data/glove-25-euclidean-workload.hdf5")


def calc_distance(a, v):
  return metrics['euclidean']['distance'](a, v);
  
distance = []
for i, v in enumerate(prev_dataset['test']):
  ans = [metrics['euclidean']['distance'](prev_dataset['train'][idx], v) for idx in prev_dataset['neighbors'][i]]
  # distance.append([metrics['euclidean']['distance'](prev_dataset['train'][idx], v) for idx in prev_dataset['neighbors'][i]])
  distance.append(ans)
  # print(ans)
  # print(prev_dataset['train'][prev_dataset['neighbors'][i][2]], v, calc_distance(prev_dataset['train'][prev_dataset['neighbors'][i][2]], v ))
  if(i % 1000 == 0): print(i)
distance = np.vstack(distance)
print(distance.shape)

del prev_dataset["distances"]
prev_dataset["distances"] = distance
prev_dataset.close()