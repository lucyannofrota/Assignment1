import random_forest
import numpy as np
from scipy import stats

a = [1, 2, 3, 4]
b = [2, 4, 2, 5]

c = []

c.append(a)
c.append(b)
c.append(b)
print(stats.mode(c, axis=0))