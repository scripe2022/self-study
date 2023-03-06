#!/usr/bin/python3
import numpy as np
from skimage import io
from collections import defaultdict
from scipy.signal import convolve
import matplotlib.pyplot as plt

base = 18.2
u = np.array([0.6, -1, 0])
v = np.array([1.5, 0.67, 0])
u = u / np.linalg.norm(u)
v = v / np.linalg.norm(v)
w = np.cross(u, v)
print(w)