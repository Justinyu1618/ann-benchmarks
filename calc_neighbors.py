from ann_benchmarks.algorithms.annoy import Annoy
from ann_benchmarks.distance import metrics
import h5py 
import numpy as np

prev_dataset = h5py.File("./data/glove-25-euclidean-workload.hdf5")
print("annoy w/ euclidean")
annoy = Annoy("euclidean", 1)
annoy.set_query_arguments(1300000)
print("about to fit")
annoy.fit(prev_dataset['train'])
print("finished fit")

neighbors = []
for i, v in enumerate(prev_dataset['test']):
  neigh = annoy.query(v, 100)
  neighbors.append(neigh)
  print(i)

neighbors = np.vstack(neighbors)
print(neighbors.shape)

del prev_dataset["neighbors"]
prev_dataset["neighbors"] = neighbors
prev_dataset.close()


# def calc_distance(a, v):
#   return metrics['angular']['distance'](a, v);
  
# distance = []
# for i, v in enumerate(prev_dataset['test']):
#   distance.append([metrics['angular']['distance'](prev_dataset['train'][idx], v) for idx in prev_dataset['newer_neighbors'][i]])
#   if(i % 1000 == 0): print(i)
# distance = np.vstack(distance)
# print(distance.shape)

# del prev_dataset["distances"]
# prev_dataset["distances"] = distance
# prev_dataset.close()