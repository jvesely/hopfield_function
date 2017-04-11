#!/bin/env python3

import numpy as np
from hopfield_func import HopfieldFunc

DOM_BITS = 36
RNG_BITS = 34
COUNT = 10

hf = HopfieldFunc(DOM_BITS, RNG_BITS)

def test(hf, a, b):
    a %= 2 ** DOM_BITS
    b %= 2 ** RNG_BITS
#    print("f(%s) = %s" % (a, b))
    hf.set(a, b)
#    print(hf(a))


keys = np.random.randint(0, 2 ** DOM_BITS, COUNT)
values = np.random.randint(0, 2 ** RNG_BITS, COUNT)

for x in map(lambda x,y: (x,y), keys, values):
    test(hf, x[0], x[1])

results = [(x[0], x[1], hf(x[0])) for x in map(lambda x,y: (x,y), keys, values)]

res = sum(x[1] == x[2] for x in results)
for x in results:
    if x[2] != x[1]:
        print("FAIL: hf(%d) = %d instead of %d" % (x[0], x[2], x[1]))
    else:
        print("PASS: hf(%d) = %d" % (x[0], x[1]))
print("Successfully remembered: %d out of %d" % (res, COUNT))
