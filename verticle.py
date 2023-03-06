#!/usr/bin/python3
import numpy as np
from skimage import io
from collections import defaultdict
from scipy.signal import convolve
import matplotlib.pyplot as plt

def rotZ(theta):
    a = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return a

def rotX(theta):
    a = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ])
    return a

def rotY(theta):
    a = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    return a

def t_x(t):
    a = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    return a

def intrinsics(d, c):
    a = np.array([
        [-d[0], 0, c[0]],
        [0, -d[1], c[1]],
        [0, 0, 1]
    ])
    return a

def projection():
    a = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    return a

def rotation(R):
    a = np.array([
        [R[0, 0], R[0, 1], R[0, 2], 0],
        [R[1, 0], R[1, 1], R[1, 2], 0],
        [R[2, 0], R[2, 1], R[2, 2], 0],
        [0, 0, 0, 1]
    ])
    return a

def translation(c):
    a = np.array([
        [1, 0, 0, -c[0]],
        [0, 1, 0, -c[1]],
        [0, 0, 1, -c[2]],
        [0, 0, 0, 1]
    ])
    return a

I = intrinsics((378, 378), (0, 0))
P = projection()
size = 100
pixel = 378
# y = np.arange(-45, 45)
y = np.array([-36.7, -17.3, 2.1, 21.5, 40.9, 2.1])
x = np.ones(y.shape) * 8.5
x[5] = 7
z = np.ones(y.shape) * (-50)
z[5] = -40
w = np.ones(y.shape)
world = np.stack((x, y, z, w), axis=1)
camera = (I @ P @ world.T).T
pp = np.array([10, 45, -50, 1]).T
print(I @ P @ pp)

camera[:, 0] = (camera[:, 0] / camera[:, 2]) + pixel
camera[:, 1] = (camera[:, 1] / camera[:, 2]) + pixel
original= io.imread('imgs/25percent/verticle.jpg')
image = io.imread('imgs/25percent/verticle.jpg', as_gray=True)
plt.imshow(image, cmap='gray')
plt.plot(camera[:, 0], camera[:, 1], 'o')
plt.show()

# lambda = 0.132275132