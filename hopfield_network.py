#!/bin/python3

import numpy as np

class HopfieldNetwork(object):
    def __init__(self, dim, iterations = 1000):
        assert dim > 0
        self.__matrix = np.zeros((dim,dim))
        self.__memories = 0
        self.__iterations = iterations

    ''' the matrix format is:
        sum{p}{outer_product(memory_i, memory_i)} /N  - (p/N)(Identity)
        that is equivalent to:
        sum{p}{(outer_product(memory_i, memory_i) - Identity)}/N
        Thus to add a new memory to M:
        M = (M + (outer_product(new_memory, new_memory) - Identity))/N
    '''
    def remember(self, data):
        # convert '0' -> '-1'
        data[data < 0.5] = -1

        # create new memory
        new_memory = np.asmatrix(np.outer(data, data)) / self.__matrix.size
        np.fill_diagonal(new_memory, 0)
        assert new_memory.shape == self.__matrix.shape

        # add the new memory to memory matrix
        self.__matrix = (self.__matrix + new_memory);
        self.__memories += 1
   
    def recall(self, partial_data, iterations = None):
        # get iterations overload
        iterations = self.__iterations if (iterations is None) else iterations
        partial_data[partial_data < 0.5] = -1
       
        # turn partial data into column vector
        probe = np.asmatrix(partial_data).transpose()
        assert partial_data.size  == self.__matrix.shape[0]

        # iterate
        # Synchronous update, should we try asynchronous?
        for i in range(iterations):
            probe = np.sign(np.matmul(self.__matrix, probe))

        # turn column vector back into array
        res = np.squeeze(np.asarray(probe.transpose()))

        # replace all '-1' with 0
        res[res < 0] = 0
        return res
