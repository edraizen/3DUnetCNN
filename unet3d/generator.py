import os
from random import shuffle

import numpy as np

#from .utils import pickle_dump, pickle_load
#from .augment import augment_data

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

class FileGenerator(DataGenerator):
    @classmethod
    def from_files(cls, files, truth_files=None, batch_size=1, data_split=0.8, num_samples=None):
        """Read in data from one more more numpy files"""
        (np.load(file) for file in files)

    @classmethod
    def get_training_and_validation(cls, data_files, truth_files, batch_size=1, data_split=0.8, num_samples=None):
        assert lens(data_files) == len(truth_files)
        for data_file, truth_file in it.izip(data_files, truth_files):
            pass
        
        train = cls()
        validation = cls()
        return train, validation

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
        if self.current > self.n_samples:
            raise StopIteration

        self.current += 1
        shape = [1]+list(self.shape)
        volume = np.zeros(shape)
        labels = np.zeros(shape)
        v, l = create_spheres(self.cnt, self.shape[1:], self.border, self.r_min, self.r_max)
        volume[0, 0, :, :, :] = v
        labels[0, 0, :, :, :] = l
        return volume, labels

def create_spheres(num_spheres, shape=(144, 144, 144), border=50, min_r=5, max_r=15):
    """Create randomly placed and randomy sized spheres inside of a grid
    """
    volume = np.random.random(shape)
    labels = np.zeros(shape)
    print border, shape
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
