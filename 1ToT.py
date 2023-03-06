#!/usr/bin/python3
import numpy as np
from skimage import io
from collections import defaultdict
from scipy.signal import convolve
import matplotlib.pyplot as plt

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

def solve_dots(l1, l2):
    A = np.array([l1[:2], l2[:2]])
    B = np.array([-l1[2], -l2[2]])
    dots = np.linalg.solve(A, B)
    return dots

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

with open('points.npy', 'rb') as f:
    X1 = np.load(f)
    X2 = np.load(f)
    X3 = np.load(f)
    Xt = np.load(f)
img1 = io.imread('imgs/25percent/camera1.jpg', as_gray=True)
img2 = io.imread('imgs/25percent/camera2.jpg', as_gray=True)
img3 = io.imread('imgs/25percent/camera3.jpg', as_gray=True)
imgt = io.imread('imgs/25percent/verticle.jpg', as_gray=True)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'b', 'g', 'r', 'c', 'm']
fig, axs = plt.subplots(1, 2)
axs[0].imshow(img1, cmap='gray')
axs[0].scatter(X1[0, :], X1[1, :], c=colors[:X1.shape[1]], s=100, marker='o')
for i in range(X1.shape[1]):
    axs[0].annotate(str(i), (X1[0, i], X1[1, i]), fontsize=14, color='tab:red')
axs[1].imshow(img2, cmap='gray')
axs[1].scatter(X2[0, :], X2[1, :], c=colors[:X2.shape[1]], s=100, marker='o')
for i in range(X2.shape[1]):
    axs[1].annotate(str(i), (X2[0, i], X2[1, i]), fontsize=14, color='tab:red')
# plt.imshow(img1, cmap='gray')
# plt.scatter(X1[0, :], X1[1, :], c=colors[:X1.shape[1]], s=50, marker='o')
# for i in range(X1.shape[1]):
#     plt.annotate(str(i), (X1[0, i], X1[1, i]), fontsize=10, color='tab:red')

# axs[2].imshow(img3, cmap='gray')
# axs[2].scatter(X3[0, :], X3[1, :], c=colors[:X3.shape[1]])
# axs[2].imshow(imgt, cmap='gray')
# axs[2].scatter(Xt[0, :], Xt[1, :], c=colors[:Xt.shape[1]])
# for i in range(Xt.shape[1]):
#     axs[2].annotate(str(i), (Xt[0, i], Xt[1, i]))
plt.show()
F_1t = fundamental_matrix(X1, Xt)
F_2t = fundamental_matrix(X2, Xt)
F = {
    '1t': F_1t,
    '2t': F_2t
}
np.save('fundamentals.npy', F)
# print(X1[:, 1] @ F_1t @ Xt[:, 1])
# print(X2[:, 10] @ F_2t @ Xt[:, 10])
p1x = np.array([67.5, 139.0, 272.0, 404.74, 546.81])
p1y = np.array([258.0, 289.3, 345.7, 408.42, 468.18])
p2x = np.array([70.1, 168.94, 340.2, 500.64, 652.3])
p2y = np.array([499.3, 483.88, 454.5, 433.53, 410.9])

P1 = np.stack((p1x, p1y, np.ones(p1x.shape[0])), axis=-1).T
P2 = np.stack((p2x, p2y, np.ones(p2x.shape[0])), axis=-1).T
l1 = np.dot(F_1t.T, P1[:, 4])
l2 = np.dot(F_2t.T, P2[:, 4])
dots = solve_dots(l1, l2)
plot_line(imgt, l1, l2, dots)
print(F_1t)