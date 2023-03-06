#!/usr/bin/python3
import numpy as np
from skimage import io
from collections import defaultdict
from scipy.signal import convolve
import matplotlib.pyplot as plt

def nccMatch(img1, img2, c1, c2, R):
    """Compute NCC given two windows.

    Args:
        img1: Image 1.
        img2: Image 2.
        c1: Center (in image coordinate) of the window in image 1.
        c2: Center (in image coordinate) of the window in image 2.
        R: R is the radius of the patch, 2 * R + 1 is the window size

    Returns:
        NCC matching score for two input windows (a scalar value).

    """
    
    win1 = img1[c1[1]-R:c1[1]+R+1, c1[0]-R:c1[0]+R+1]
    win2 = img2[c2[1]-R:c2[1]+R+1, c2[0]-R:c2[0]+R+1]
    win1 = (win1 - np.mean(win1)) / np.sqrt(np.sum(np.square(win1 - np.mean(win1))))
    win2 = (win2 - np.mean(win2)) / np.sqrt(np.sum(np.square(win2 - np.mean(win2))))
    matching_score = np.sum(win1 * win2)

    return matching_score

import numpy as np
f1 = open("points1.npy", "rb")
p1 = np.load(f1)
f2 = open("points2.npy", "rb")
p2 = np.load(f2)

print(p1.shape)
img1 = io.imread('imgs/25percent/camera1.jpg')
img2 = io.imread('imgs/25percent/camera2.jpg')
R = 5
for i in range(p1.shape[0]):
    print(i)
    minScore = 0
    mini = 0
    minj = 0
    for j in range(p2.shape[0]):
        score = nccMatch(img1, img2, p1[i], p2[j], R)
        if (score > maxScore):
            maxScore = score
            maxi = i
            maxj = j