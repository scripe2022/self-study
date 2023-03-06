#!/usr/bin/python3

import numpy as np
from skimage import io
from collections import defaultdict
from scipy.signal import convolve
import matplotlib.pyplot as plt

img1 = io.imread('imgs/25percent/camera1.jpg', as_gray=True)
img2 = io.imread('imgs/25percent/camera2.jpg', as_gray=True)
img3 = io.imread('imgs/25percent/camera3.jpg', as_gray=True)
imgt = io.imread('imgs/25percent/verticle.jpg', as_gray=True)
F = np.load('fundamentals.npy', allow_pickle=True).item()

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
R = rotation(rotZ(-1.15) @ rotY(0.3) @ rotX(0.2))
T = translation((0, 0, 0))
y = np.array([-36.7, -17.3, 2.1, 21.5, 40.9, 2.1])
x = np.ones(y.shape) * 8.5
x[5] = 7
z = np.ones(y.shape) * (-50)
z[5] = -40
w = np.ones(y.shape)
world = np.stack((x, y, z, w), axis=1).T
# print(I.shape)
# print(P.shape)
# print(R.shape)
# print(T.shape)
# print(world.shape)
camera = I @ P @ R @ T @ world
camera[0, :] = (camera[0, :] / camera[2, :]) + 378
camera[1, :] = (camera[1, :] / camera[2, :]) + 378
camera = camera[:2, :]
plt.scatter(camera[0, :], camera[1, :])
plt.imshow(img1, cmap='gray')
plt.show()