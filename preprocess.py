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
    gk = gaussian2d(filter_size=9, sig=5)
    smooth_image = convolve(image, gk, mode='same')
    
    return smooth_image

def outer(a, n):
    """
    Zero padding outer n
    """
    f = np.zeros(a.shape)
    f[n:f.shape[0] - n, n:f.shape[1] - n] = a[n:f.shape[0] - n, n:f.shape[1] - n]
    return f

def normalize(a):
    a = a - np.min(a)
    a = outer(a, 3)
    a = a / np.max(a)
    return a

def gradient(image):
    kx = np.array([
        [0, 0, 0],
        [1/2, 0, -1/2],
        [0, 0, 0]
    ])
    ky = np.array([
        [0, 1/2, 0],
        [0, 0, 0,],
        [0, -1/2, 0]
    ])
    dx = convolve(image, kx, mode='same')
    dy = convolve(image, ky, mode='same')
    dx = outer(dx, 10)
    dy = outer(dy, 10)
    g_mag = np.sqrt(np.square(dx) + np.square(dy))
    g_theta = np.arctan2(dy, dx)
    
    dx = normalize(dx)
    dy = normalize(dy)
    g_mag = normalize(g_mag)
    # return dx, dy
    return g_mag, g_theta

def get_points(g_mag, th):
    points = []
    for i in range(g_mag.shape[0]):
        for j in range(g_mag.shape[1]):
            if g_mag[i, j] > th:
                points.append([i, j])
    return points

def plots(dx, img, p):
    f, axs = plt.subplots(1, 3);
    axs[0].imshow(img)
    axs[1].imshow(dx, cmap='gray')
    x = [item[0] for item in p]
    y = [item[1] for item in p]
    axs[2].imshow(img, cmap='gray')
    axs[2].scatter(y, x, c='r')
    plt.show()

def intrinsics(d):
    a = np.array([
        [-d, 0, 0],
        [0, -d, 0],
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

def trans(cx, cy, cz):
    a = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 1, -cz],
        [0, 0, 0, 1]
    ])
    return a

def get_largest(a, n):
    """
    get largest n indices in matrix a
    """
    ny = a.shape[1]
    a = a.reshape(-1)
    l = np.argpartition(-a, n)[:n]
    f = np.stack((l % ny, l // ny), axis=-1)
    return f

def pre_process(image, windowSize, smoothSTD):
    """
    1. smooth
    2. calculate dx & dy
    3. zero padding outer
    """
    gk = gaussian2d(filter_size=windowSize, sig=smoothSTD)
    smooth_image = convolve(image, gk, mode='same')
    # smooth_image = smooth_image / 255.
    kx = np.array([
        [0, 0, 0],
        [1/2, 0, -1/2],
        [0, 0, 0]
    ])
    ky = np.array([
        [0, 1/2, 0],
        [0, 0, 0,],
        [0, -1/2, 0]
    ])
    dx = convolve(smooth_image, kx, mode='same')
    dx = outer(dx, windowSize//2)
    dy = convolve(smooth_image, ky, mode='same')
    dy = outer(dy, windowSize//2)
    return dx, dy

def corner_detect(image, nCorners, smoothSTD, windowSize):
    """Detect corners on a given image.

    Args:
        image: Given a grayscale image on which to detect corners.
        nCorners: Total number of corners to be extracted.
        smoothSTD: Standard deviation of the Gaussian smoothing kernel.
        windowSize: Window size for Gaussian smoothing kernel, corner detector, and non maximum suppression.

    Returns:
        Detected corners (in image coordinate) in a numpy array (n*2).
        The minor eigen value image having the same shape as the image
    """
    dx, dy = pre_process(image, windowSize, smoothSTD)
    minor_eig_image = np.zeros_like(image)
    for i in range(windowSize//2, image.shape[0] - windowSize//2):
        for j in range(windowSize//2, image.shape[1] - windowSize//2):
            dx_win = dx[i - windowSize//2:i + windowSize//2 + 1, j - windowSize//2:j + windowSize//2 + 1]
            dy_win = dy[i - windowSize//2:i + windowSize//2 + 1, j - windowSize//2:j + windowSize//2 + 1]
            C = np.array([
                [np.sum(np.square(dx_win)), np.sum(dx_win * dy_win)],
                [np.sum(dx_win * dy_win), np.sum(np.square(dy_win))]
            ])
            eigen = np.linalg.eig(C)[0]
            minor_eig_image[i, j] = np.min(eigen)
    J = np.zeros_like(minor_eig_image)
    for i in range(windowSize//2, image.shape[0] - windowSize//2):
        for j in range(windowSize//2, image.shape[1] - windowSize//2):
            win = minor_eig_image[i - windowSize//2:i + windowSize//2 + 1, j - windowSize//2:j + windowSize//2 + 1]
            if (minor_eig_image[i, j] == np.max(win)):
                J[i, j] = minor_eig_image[i, j]
    corners = get_largest(J, nCorners)



    return corners, minor_eig_image

def nccMatch(img1, img2, c1, c2, R):
    win1 = img1[c1[1]-R:c1[1]+R+1, c1[0]-R:c1[0]+R+1]
    win2 = img2[c2[1]-R:c2[1]+R+1, c2[0]-R:c2[0]+R+1]
    win1 = (win1 - np.mean(win1)) / np.sqrt(np.sum(np.square(win1 - np.mean(win1))))
    win2 = (win2 - np.mean(win2)) / np.sqrt(np.sum(np.square(win2 - np.mean(win2))))
    matching_score = np.sum(win1 * win2)

    return matching_score

original= io.imread('imgs/25percent/camera2.jpg')
image = io.imread('imgs/25percent/camera2.jpg', as_gray=True)
image = smooth(image)
g_mag, g_theta = gradient(image)
# dx, dy = gradient(image)
print(np.max(g_mag))
print(np.min(g_mag))
p = get_points(g_mag, 0.5)
np.save('points2', p)
# p, me = corner_detect(image, 1000, 1.0, 9)
print(len(p))
plots(g_mag, original, p)
I = intrinsics(756)
P = projection()
T = trans(60, 35, -5)

def try_plot(image, p):
    plt.imshow(image, cmap='gray')
    plt.scatter(p[:, 1] / p[:, 2], p[:, 0] / p[:, 2], c='r')
    plt.show()

world = np.zeros((10, 4))
for i in range(10):
    world[i, 0] = 50
    world[i, 1] = ((-i * 7) / 70.0) + 35
    world[i, 2] = 0
    world[i, 3] = 1
camera = np.zeros((10, 3))
for i in range(10):
    # world[i] = T @ world[i]
    # print(world[i])
    # print(I @ P @ world[i])
    camera[i] = ((I @ P @ T @ (world[i].reshape(4, 1))).reshape(-1))
# try_plot(image, camera)
# print(camera)