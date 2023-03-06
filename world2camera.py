#!/usr/bin/python3
import numpy as np
from skimage import io
from collections import defaultdict
from scipy.signal import convolve
import matplotlib.pyplot as plt

def gaussian2d(filter_size=9, sig=1.0):
    ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)


def smooth(image):
    gk = gaussian2d(filter_size=9, sig=1.4)
    smooth_image = convolve(image, gk, mode='same')
    
    return smooth_image

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

original= io.imread('imgs/25percent/camera1.jpg')
image = io.imread('imgs/25percent/camera1.jpg', as_gray=True)
image = smooth(image)

def try_plot(image, p):
    plt.imshow(image, cmap='gray')
    plt.scatter(p[:, 0], p[:, 1], c='r')
    plt.show()

I = intrinsics((756, 756), (0, 0))
P = projection()
rd = np.array([-0.8, 0.4, 0])
Rx = rotX(rd[0])
Ry = rotY(rd[1])
Rz = rotZ(rd[2])
R = rotation(Rz @ Ry @ Rx)
T = translation(np.array([47, 40, 5]))
world = np.zeros((10, 4))
for i in range(10):
    world[i, 0] = 50
    world[i, 1] = ((i * 7) / 15.0) + 35
    world[i, 2] = 0
    world[i, 3] = 1
# camera = np.zeros((10, 3))
# for i in range(10):
#     camera[i] = I @ P @ R @ T @ (world[i].reshape(-1))
camera = (I @ P @ R @ T @ world.T).T
print(camera)
camera[:, 0] /= camera[:, 2]
camera[:, 1] /= camera[:, 2]
camera = camera[:, :2]
try_plot(image, camera)