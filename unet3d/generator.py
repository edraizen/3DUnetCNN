import os
from random import shuffle

import numpy as np
import h5py

import itertools as it

class DataGenerator(object):
    def __init__(self, data, truth=None, batch_size=1, indexes=None, shape=None, augment=False, train=True):
        self.data = data
        self.truth = truth
        self.batch_size = batch_size
        self.shape = shape
        self.augment = augment
        self.num_steps = len(indexes) if indexes else data.shape[0]
        if train:
            self.num_steps /= float(self.batch_size)
        self.indexes = iter(suffle(indexes)) or iter(shuffle(xrange(data.shape[0])))
        self.it = xrange(0, len(indexes) if indexes else data.shape[0], self.batch_size)

    @classmethod
    def get_training_and_validation(cls, data, truth, batch_size=1, data_split=0.8, shuffle_list=True):
        n_training = int(data.shape[0] * data_split)
        indexes = range(data.shape[0])
        if shuffle_list:
            shuffle(indexes)
        
        train = cls(data, truth, batch_size=batch_size, indexes=indexes[:n_training])
        validation = cls(data_file.root.data, data_file.truth.data, batch_size=batch_size, indexes=indexes[n_training:], train=False)
        return train, validation

    def __iter__(self):
        return self

    def next(self):
        index = self.it.next() #Will raise StopIteration
        indexes = self.indexes[index:index+self.batch_size]
        data = self.data[indexes]
        if self.truth:
            truth =  self.truth[indexes]
        else:
            truth = None

        if self.augment and False:
            data, truth = augment_data(data, truth, affine, flip=augment_flip,
                                       scale_deviation=augment_distortion_factor)

        return x, y

class DistributedH5FileGenerator(object):
    def __init__(self, files, batch_size=1, train=True):
        self.files = files
        self.batch_size = batch_size
        self.num_steps = len(files)
        if train:
            self.num_steps /= float(self.batch_size)
        self.current = 0
        self.indexes = range(len(files))
        self.train = train

    @classmethod
    def get_training_and_validation(cls, files, batch_size=1, data_split=0.8, num_samples=None, shuffle_list=True):
        n_training = int(len(files) * data_split)
        files = np.array(files)
        indexes = range(len(files))
        if shuffle_list:
            shuffle(indexes)
	indexes = np.array(indexes)
        train = cls(files[indexes[:n_training]], batch_size=batch_size)
        validation = cls(files[indexes[n_training:]], batch_size=batch_size, train=False)
        return train, validation

    def __iter__(self):
        return self

    def update(self):
        print "Updating"
        self.indexes = shuffle(self.indexes)
        self.current = 0
        print "index is now", self.current

    def next(self): 
       print self.current, self.indexes[self.current], self.files[self.indexes[self.current]]
       with h5py.File(os.path.abspath(self.files[self.indexes[self.current]]), "r") as f:
		data = f["data"][()][np.newaxis]
		truth = f["truth"][()][np.newaxis]
       print self.train, self.current, len(self.indexes)-1
       if self.current < len(self.indexes)-1:
           self.current += 1
       else:
           self.update()
       return data, truth

class FileGenerator(object):
    def __init__(self, data, truth, n_samples, batch_size=1):
        self.data = data
        self.current = 0
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.current = 0
        self.num_steps = n_samples
        if train:
            self.num_steps /= float(self.batch_size)

    @classmethod
    def get_training_and_validation(cls, data_files, truth_files, batch_size=1, data_split=0.8, num_samples=None, num_samples_per_class=None, num_classes_per_file=None):
        """Read in data from one more more numpy files"""
        data_files = [np.load(file, mmap_mode="r") for file in data_files]
        truth_files = [np.load(file, mmap_mode="r") for file in truth_files]

        if num_classes_per_file is None:
            assert 0, "Not implemented, must choose 2"

        if num_samples_per_class is None:
            assert 0, "Not implemented, must choose 20"

        train_num = num_samples_per_class*data_split
        validation_num = num_samples_per_class-train_num

        train = (cls(
            f[:train_num],
            t[:train_num],
            n_samples=train_num, 
            batch_size=batch_size) for f, t in it.izip(data_files, truth_files))

        validation = (cls(
            f[train_num:], 
            t[train_num:],
            n_samples=validation_num, 
            batch_size=batch_size) for f, t in it.izip(data_files, truth_files))

        train_gen = it.chain(iter(t) for t in train)
        validation_gen = it.chain(iter(v) for v in validation)

        return train, validation, train_gen, validation_gen

    def __iter__(self):
        return self

    def next(self):
        data = self.data[self.current, ...]
        self.current += 1
        return data

class NiftyFileUNet3d(DataGenerator):
    @classmethod
    def get_training_and_validation(cls, data_file, batch_size=1, data_split=0.8, shuffle_list=True):
        train, validation = DataGenerator.get_training_and_validation(
            data_file.root.data, data_file.root.truth.data,
            batch_size=batch_size, data_split=data_split, shuffle_list=shuffle_list)
        return train, validation

class ExampleSphereGenerator(object):
    """Create spheres surrounded by noise
    """
    def __init__(self, shape, cnt=10, r_min=10, r_max=50, border=92, sigma=20, batch_size=1, n_samples=100, train=True):
        self.shape = shape
        self.cnt = cnt
        self.r_min = r_min
        self.r_max = r_max
        self.border = border
        self.sigma = sigma
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.current = 0
        self.num_steps = n_samples
        if train:
            self.num_steps /= float(self.batch_size)

    @classmethod
    def get_training_and_validation(cls, shape, cnt=10, r_min=10, r_max=30, border=50, batch_size=1, n_samples=100, data_split=0.8):
        train = cls(shape, cnt=cnt, r_min=r_min, r_max=r_max, border=border, batch_size=batch_size, n_samples=n_samples*data_split)
        validate = cls(shape, cnt=cnt, r_min=r_min, r_max=r_max, border=border, batch_size=batch_size, n_samples=n_samples*(1-data_split))
        return train, validate

    def __iter__(self):
        # Iterators are iterables too.
        # Adding this functions to make them so.
        return self
        
    def next(self):
        #if self.current > self.n_samples:
        #    raise StopIteration

        self.current += 1
        shape = [1]+list(self.shape)
        volume = np.zeros(shape)
        labels = np.zeros(shape[:-1]+[1])
        print volume.shape
        v, l = create_spheres(self.cnt, self.shape[:-1], self.border, self.r_min, self.r_max)
        volume[0, :, :, :, 0] = v
        labels[0, :, :, :, :] = l
        print volume.shape, labels.shape
        return volume, labels

def create_spheres(num_spheres, shape=(144, 144, 144), border=50, min_r=5, max_r=15):
    """Create randomly placed and randomy sized spheres inside of a grid
    """
    volume = np.random.random(shape)
    labels = np.zeros(list(shape)+[1])
    print border, shape, map(lambda i: i-border, shape)
    for i in xrange(num_spheres):
        #Define random center of sphere and radius
        center = [np.random.randint(border, edge-border) for edge in shape]
        r = np.random.randint(min_r, max_r)
        color = np.random.random()

        y, x, z = np.ogrid[-center[0]:shape[0]-center[0], -center[1]:shape[1]-center[1], -center[2]:shape[2]-center[2]]
        m = x*x + y*y + z*z < r*r
        indices = np.where(m==True)
        volume[indices] = color
        labels[indices] = 1

    return volume, labels
