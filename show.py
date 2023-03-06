#!/usr/bin/python3
import numpy as np
from skimage import io
from collections import defaultdict
from scipy.signal import convolve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from matplotlib.patches import Rectangle
from scipy.io import loadmat
from numpy.linalg import svd

def compute_fundamental(x1, x2):
    """ Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the 8 point algorithm.
        
        Construct the A matrix according to lecture
        and solve the system of equations for the entries of the fundamental matrix.

        Returns:
        Fundamental Matrix (3x3)
    """
    
    x1, x2 = x2, x1
    A = np.stack((x2[0,:]*x1[0,:], x2[0,:]*x1[1,:], x2[0,:], x2[1,:]*x1[0,:], x2[1,:]*x1[1,:], x2[1,:], x1[0,:], x1[1,:], np.ones(x1.shape[1])), axis=-1)
    U, D, V = np.linalg.svd(A)
    F = V.T[:, 8].reshape(3, 3)
    
        
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))
    
    return F/F[2,2]

def fundamental_matrix(x1,x2):
    # Normalization of the corner points is handled here
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = np.dot(T1,x1)
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = np.dot(T2,x2)

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = np.dot(T1.T,np.dot(F,T2))

    return F/F[2,2]

def plot_epipolar_lines(img1, img2, cor1, cor2):
    """Plot epipolar lines on image given image, corners

    Args:
        img1: Image 1.
        img2: Image 2.
        cor1: Corners in homogeneous image coordinate in image 1 (3xn)
        cor2: Corners in homogeneous image coordinate in image 2 (3xn)

    """
    
    assert cor1.shape[0] == 3
    assert cor2.shape[0] == 3
    assert cor1.shape == cor2.shape
    
    F = fundamental_matrix(cor1, cor2)
        
    # epipole in image 1 is the solution to F^T e = 0
    U,S,V = np.linalg.svd(F.T)
    e1 = V[-1]
    e1 /= e1[-1]
    
    # epipole in image 2 is the solution to Fe = 0
    U,S,V = np.linalg.svd(F)
    e2 = V[-1]
    e2 /= e2[-1]

    plot_epipoles = False
    
    # Plot epipolar lines in the first image
    # There is an epipolar line for each corner
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(img1, cmap='gray')
    h, w = img1.shape[:2]
    for ct in cor2.T:
        # epipolar line is (F * ct) dot (x, y, 1) = 0
        epi_line = np.dot(F, ct)
        a, b, c = epi_line # ax + by + c = 0, y = -a/b * x - c/b
        x = np.arange(w)
        y = (-a / b) * x - (c / b)
        x = np.array([x[i] for i in range(x.size) if y[i] >=0 and y[i] < h - 1])
        y = np.array([y[i] for i in range(y.size) if y[i] >=0 and y[i] < h - 1])
        plt.plot(x, y, 'b', zorder=1)
        
    plt.scatter(cor1[0], cor1[1], s=50, edgecolors='b', facecolors='r', zorder=2)
    
    if plot_epipoles:
        plt.scatter([e1[0]], [e1[1]], s=75, edgecolors='g', facecolors='y', zorder=3)
    plt.show()
    
    # Plot epipolar lines in the second image
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(img2, cmap='gray')
    h, w = img2.shape[:2]
    
    for c1 in cor1.T:
        # epipolar line is (F^T * c1) dot (x, y, 1) = 0
        epi_line = np.dot(F.T, c1)
        a, b, c = epi_line
        x = np.arange(w)
        y = (-a / b) * x - (c / b)
        x = np.array([x[i] for i in range(x.size) if y[i] >=0 and y[i] < h - 1])
        y = np.array([y[i] for i in range(y.size) if y[i] >=0 and y[i] < h - 1])
        plt.plot(x, y, 'b', zorder=1)
    
    plt.scatter(cor2[0], cor2[1], s=50, edgecolors='b', facecolors='r', zorder=2)
    
    if plot_epipoles:
        plt.scatter([e2[0]], [e2[1]], s=75, edgecolors='g', facecolors='y', zorder=3)
    plt.show()

img1 = io.imread('imgs/25percent/camera1.jpg', as_gray=True)
img2 = io.imread('imgs/25percent/camera2.jpg', as_gray=True)
with open('points.npy', 'rb') as f:
    X1 = np.load(f)
    X2 = np.load(f)
    X3 = np.load(f)
    Xt = np.load(f)
plot_epipolar_lines(img1, img2, X1, X2)