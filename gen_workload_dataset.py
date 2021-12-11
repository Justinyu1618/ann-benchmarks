import numpy as np
import csv 
import h5py
import time, pickle
from ann_benchmarks.algorithms.annoy import Annoy

def read_txt_data(filename):
  txt_labels = np.genfromtxt(filename, dtype="str", invalid_raise=False, usecols=[0], converters={0: lambda x: x.decode()})
  data = np.genfromtxt(filename, invalid_raise=False)
  data = data[:, 1:]
  return txt_labels, data


prev_dataset = h5py.File("./data/glove-25-euclidean-workload.hdf5")

labels, data = read_txt_data("./data/glove.twitter.27B/glove.twitter.27B.25d.txt")
labels_to_i = {label: i for i, label in enumerate(labels)}

pickle.dump(labels,open("./data/glove-labels.pkl", "wb"))

del prev_dataset['train']
prev_dataset['train'] = data
prev_dataset.close()


word_freqs = {}
word_probs = []
with open("./unigram_freq.csv", "r") as f:
  reader = csv.reader(f)
  next(reader)
  word_freqs = {word: int(freq) for word, freq in reader}

pickle.dump(word_freqs,open("./data/word_freqs.pkl", "wb"))


# # print(labels)
# filtered_labels = [label for label in labels if label in word_freqs]
# label_freqs = [word_freqs[word] for word in filtered_labels]
# word_probs = np.array(label_freqs) / sum(label_freqs)



# def sample(num):
#   return data[np.array([labels_to_i[label] for label in np.random.choice(filtered_labels, size=num, p=word_probs)])]

# test_data_len = prev_dataset['test'].shape[0]

# new_test_data = sample(test_data_len)
# print(prev_dataset['test'].shape, new_test_data.shape)


# del prev_dataset['test']
# prev_dataset['test'] = new_test_data
# prev_dataset.close()


