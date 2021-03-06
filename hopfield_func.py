#!/bin/env python3

import sys
import numpy as np
from hopfield_network import HopfieldNetwork

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class HopfieldFunc(HopfieldNetwork):
    def __init__(self, domain_bits, range_bits, iterations = 100, pad_random = True, preserve = False, storage = np.inf):
        assert domain_bits > 0 and domain_bits < 48
        assert range_bits > 0 and range_bits < 48
        self.__domain_bits = domain_bits
        self.__range_bits = range_bits
        #using super() does not work for some reason
        self.__network = HopfieldNetwork(domain_bits + range_bits, iterations, preserve_elements = 0 if not preserve else domain_bits, clip = storage / 2)
        self.__pad_random = pad_random

    def set(self, x, y):
        bin_x = np.array([int(x) for x in bin(x)[2:]])
        bin_y = np.array([int(y) for y in bin(y)[2:]])
        self.__network.remember(np. concatenate(( \
                      np.zeros(self.__domain_bits - bin_x.size), bin_x,
                      np.zeros(self.__range_bits - bin_y.size), bin_y)))

    def get(self, x):
        bin_x = np.array([int(x) for x in bin(x)[2:]])
        padding = np.random.randint(0, 1, self.__range_bits) if self.__pad_random else np.zeros(self.__range_bits)

        memory = self.__network.recall(np. concatenate(( \
                    np.zeros(self.__domain_bits - bin_x.size), bin_x, padding)))
        bin_res = memory[self.__domain_bits:]
        assert(bin_res.size == self.__range_bits)
        int_res = int(''.join(str(int(x)) for x in bin_res), 2)

        bin_x = memory[:self.__domain_bits]
        assert(bin_x.size == self.__domain_bits)
        int_x = int(''.join(str(int(x)) for x in bin_x), 2)
        if (int_x != x):
            eprint("Warning input changed: %x -> %x" % (x, int_x))
        return int_res

    def __call__(self, x):
        return self.get(x);
