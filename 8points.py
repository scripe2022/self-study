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

c1 = io.imread('imgs/25percent/camera1.jpg')
ct = io.imread('imgs/25percent/verticle.jpg')
c2 = io.imread('imgs/25percent/camera2.jpg')
c3 = io.imread('imgs/25percent/camera3.jpg')
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r']
c1_x = np.array([365.7, 405.1, 9.72, 482.4, 363.84, 271.2, 139.35, 420.4, 457.0, 372.5, 464.7, 474.4, 473.7, 546.6, 474.4, 463.8, 412.8, 424.2, 455.0, 480.6, 507.9, 534.3])
c1_y = np.array([498.3, 408.8, 469.79, 231.6, 184.68, 346.3, 289.52, 617.7, 435.4, 700.8, 446.5, 459.6, 419.4, 470.4, 597.9, 520.4, 435.9, 384.8, 376.9, 400.6, 434.1, 484.3])
c2_x = np.array([542.3, 500.6, 190.26, 424.66, 293.3, 339.81, 168.62, 682.80, 608.51, 716.5, 629.2, 651.7, 628.0, 653.2, 722.7, 660.1, 550.9, 529.8, 549.2, 590.6, 648.7, 703.3])
c2_y = np.array([530.1, 433.4, 745.21, 259.62, 277.1, 454.48, 484.23, 570.99, 373.06, 696.6, 374.0, 376.1, 346.4, 409.7, 487.8, 443.8, 409.5, 359.3, 337.2, 340.8, 348.7, 372.5])
ct_x = np.array([341.25, 439, 231, 695.36, 690.87, 441.81, 440.15, 289.05, 382.2, 194.7, 370.8, 353.3, 369.2, 442.2, 304.2, 329.8, 387.0, 424.9, 455.7, 452.2, 412.6, 394.1])
ct_y = np.array([395.2, 391, 116.8, 384.13, 235.28, 248.13, 102.66, 462.36, 437.0, 466.4, 448.6, 462.1, 445.9, 535.5, 494.9, 468.5, 404.2, 385.7, 406.8, 438.5, 493.1, 508.1])
c3_x = np.array([403.16, 278.36, 702.16, -13.4, 17.04, 323.7, 382.1, 435.4, 281.27])
c3_y = np.array([318.95, 366.13, 669.54, 476.2, 662.76, 554.9, 772.3, 203.0, 243.74])
X1 = np.stack((c1_x, c1_y, np.ones(c1_x.shape[0])), axis=-1).T
Xt = np.stack((ct_x, ct_y, np.ones(ct_x.shape[0])), axis=-1).T
X2 = np.stack((c2_x, c2_y, np.ones(c2_x.shape[0])), axis=-1).T
X3 = np.stack((c3_x, c3_y, np.ones(c3_x.shape[0])), axis=-1).T
with open('points.npy', 'wb') as f:
    np.save(f, X1)
    np.save(f, X2)
    np.save(f, X3)
    np.save(f, Xt)
# F = fundamental_matrix(x1,x2)
# print(F)
# print(x1[:, 0] @ F @ x2[:, 0])
# plot_epipolar_lines(c1, ct, x1, x2)

# f, axs = plt.subplots(1, 4)
# axs[0].imshow(ct)
# axs[0].scatter(ct_x, ct_y, c=colors[:9])
# axs[1].imshow(c1)
# axs[1].scatter(c1_x, c1_y, c=colors[:9])
# axs[2].imshow(c2)
# axs[2].scatter(c2_x, c2_y, c=colors[:9])
# axs[3].imshow(c3)
# axs[3].scatter(c3_x, c3_y, c=colors[:9])

def plot_line(img, a, b, d):
    # a[0]x + a[1]y = a[2]
    x = np.arange(0, 762)
    y = ((-a[0] / a[1]) * x) - (a[2] / a[1])
    plt.plot(x, y, linestyle='solid')
    x = np.arange(0, 762)
    y = ((-b[0] / b[1]) * x) - (b[2] / b[1])
    plt.plot(x, y, linestyle='solid')
    plt.scatter([dots[0]], [dots[1]], s=75)
    plt.imshow(img, cmap='gray')
    plt.show()

# plt.show()
p1 = x1[:, 2]
F12 = fundamental_matrix(x1, x2)
# p2 = np.array([396, 355, 1])
# p2 = np.array([x2[1, 0], x2[0, 0], x2[2, 0]])
p2 = x2[:, 2]
F13 = fundamental_matrix(x1, x3)
# p3 = np.array([153, 514, 1])
# p3 = np.array([x3[1, 0], x3[0, 0], x3[2, 0]])
p3 = x3[:, 2]
print(p2)
print(p3)

print(p1.T @ F12 @ p2)
print(p1.T @ F13 @ p3)

l2 = np.dot(F12, p2)
l3 = np.dot(F13, p3)
print(l2)
A = np.array([l2[:2], l3[:2]])
B = np.array([-l2[2], -l3[2]])
dots = np.linalg.solve(A, B)
# plot_line(c1, l2)
plot_line(c1, l2, l3, dots)