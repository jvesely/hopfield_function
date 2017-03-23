#!/bin/python3

import numpy as np
from hopfield_network import HopfieldNetwork

class HopfieldFunc(HopfieldNetwork):
    def __init__(self, domain_bits, range_bits, iterations = 100):
        assert domain_bits > 0 and domain_bits < 48
        assert range_bits > 0 and range_bits < 48
        self.__domain_bits = domain_bits
        self.__range_bits = range_bits
        #using super() does not work for some reason
        self.__network = HopfieldNetwork(domain_bits + range_bits, iterations)

    def set(self, x, y):
        bin_x = np.array([int(x) for x in bin(x)[2:]])
        bin_y = np.array([int(y) for y in bin(y)[2:]])
        self.__network.remember(np. concatenate(( \
                      np.zeros(self.__domain_bits - bin_x.size), \
                      bin_x, \
                      np.zeros(self.__range_bits - bin_y.size), bin_y)))

    def get(self, x):
        bin_x = np.array([int(x) for x in bin(x)[2:]])
        memory = self.__network.recall(np. concatenate(( \
                    np.zeros(self.__domain_bits - bin_x.size), bin_x, \
                    np.zeros(self.__range_bits))))
#        print(memory)
        bin_res = memory[self.__range_bits:]
        return int(''.join(str(int(x)) for x in bin_res), 2)

    def __call__(self, x):
        return self.get(x);
