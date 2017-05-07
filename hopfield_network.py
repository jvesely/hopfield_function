#!/bin/env python3

import numpy as np

class HopfieldNetwork(object):
    async_default = False
    iterations_default = 1000
    def __init__(self, dim, iterations = iterations_default, async = async_default, hoppfield_original = False, preserve_elements = 0, clip = np.inf):
        assert dim > 0
        self.__matrix = np.zeros((dim,dim))
        self.__iterations = iterations
        self.__async = async
        self.__hoppfield_original = hoppfield_original
        self.__preserve_elements = preserve_elements
        self.__clip = clip

    ''' the matrix format is:
        sum{p}{outer_product(memory_i, memory_i)} /N  - (p/N)(Identity)
        that is equivalent to:
        sum{p}{(outer_product(memory_i, memory_i) - Identity)}/N
        Thus to add a new memory to M:
        M = (M + (outer_product(new_memory, new_memory) - Identity))/N
    '''
    def remember(self, data):
        # convert '0' -> '-1'
        data[data <= 0] = -1

        # The original Hoppfield paper does not normalize the weights
        # to number of neurons
        scale = 1 if self.__hoppfield_original else self.__matrix.size

        # create new memory
        new_memory = np.asmatrix(np.outer(data, data)) / scale
        np.fill_diagonal(new_memory, 0)

        assert new_memory.shape == self.__matrix.shape

        # add the new memory to memory matrix
        self.__matrix = (self.__matrix + new_memory);

        # Clip before replacing preserve lines. Clipping won't impact diagonal
        # in standard lines, but it might impact diagonal of preserved lines
        # which gets fixed below, by replacing the lines
        m = self.__matrix.clip(-self.__clip * scale, self.__clip * scale)
        assert np.isfinite(self.__clip) or (m == self.__matrix).all()

        # The above addition on messed with either non-preserved lines,
        # or '0.0' elements in the preserved lines.
        # Force replacing with eye matrix every time fixes this
        if self.__preserve_elements:
            # This creates __preserve_elements x data.size matrix with ones
            # on the diagonal
            m = np.eye(self.__preserve_elements, data.size)
            self.__matrix.put(np.arange(m.size),m.flatten());
   
    def recall(self, partial_data, iterations = None):
        assert partial_data.size  == self.__matrix.shape[0]

        # get iterations overload
        iterations = self.__iterations if (iterations is None) else iterations
        # The original paper uses {0,1} for neuronal representation
        if (not self.__hoppfield_original):
            partial_data[partial_data < 0.5] = -1
       
        # turn partial data into column vector
        probe = np.asarray(partial_data) if self.__async else np.asmatrix(partial_data).transpose()

        # iterate
        for i in range(iterations):
            if self.__async:
                # Use random async
                for v in np.random.permutation(probe.size):
                    probe[v] = np.sign(np.dot(self.__matrix[v], probe))
            else:
                probe = np.sign(np.matmul(self.__matrix, probe))
            # The original paper uses {0,1} for neuronal states
            if (self.__hoppfield_original):
                probe[probe <= 0] = 0

        # turn column vector back into array
        res = np.squeeze(np.asarray(probe.transpose()))

        # replace all '-1' with 0
        res[res < 0] = 0
        return res
