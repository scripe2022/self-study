#!/usr/bin/python3

import numpy as np
from skimage import io
from collections import defaultdict
from scipy.signal import convolve
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def solve_dots(l1, l2):
    A = np.array([l1[:2], l2[:2]])
    B = np.array([-l1[2], -l2[2]])
    dots = np.linalg.solve(A, B)
    return dots

F = np.load('fundamentals.npy', allow_pickle=True).item()
with open('points.npy', 'rb') as f:
    X1 = np.load(f)
    X2 = np.load(f)
    X3 = np.load(f)
    Xt = np.load(f)
img1 = io.imread('imgs/25percent/camera1.jpg', as_gray=True)
img2 = io.imread('imgs/25percent/camera2.jpg', as_gray=True)
img3 = io.imread('imgs/25percent/camera3.jpg', as_gray=True)
imgt = io.imread('imgs/25percent/verticle.jpg', as_gray=True)
# fig, axs = plt.subplots(1, 3)
# axs[0].imshow(img1, cmap='gray')
# axs[1].imshow(img2, cmap='gray')
# # axs[2].imshow(img3, cmap='gray')
# axs[2].imshow(imgt, cmap='gray')
# plt.show()

def plot_line(axs, P1, P2):
    l1 = F.get('1t').T @ P1
    l2 = F.get('2t').T @ P2
    result = np.zeros((2, l1.shape[1]))
    for i in range(l1.shape[1]):
        dots = solve_dots(l1[:, i], l2[:, i])
        result[:, i] = dots
    print(result)
    # plt.imshow(imgt, cmap='gray')
    axs.scatter(result[0, :]-55, result[1, :], s=200, marker='x')
    axs.plot(result[0, :]-55, result[1, :], linewidth=5)
    # for i in range(result.shape[1]-1):
    #     plt.plot(result[0, i], result[1, i+1])

# p1x = np.array([67.5, 139.0, 272.0, 404.74, 546.81])
# p1y = np.array([258.0, 289.3, 345.7, 408.42, 468.18])
# p2x = np.array([70.1, 168.94, 340.2, 500.64, 652.3])
# p2y = np.array([499.3, 483.88, 454.5, 433.53, 410.9])

p1x = np.array([65.5, 140.7, 276.0, 405.74, 542.81])
p1y = np.array([258.0, 289.3, 345.7, 408.42, 468.18])
p2x = np.array([70.1, 168.94, 340.2, 500.64, 652.3])
p2y = np.array([499.3, 483.88, 454.5, 433.53, 410.9])


s1x = np.array([474.4, 463.8, 412.8, 424.2, 455.0, 480.6, 507.9, 534.3])
s1y = np.array([597.9, 520.4, 435.9, 384.8, 376.9, 400.6, 434.1, 484.3])
s2x = np.array([722.7, 660.1, 550.9, 529.8, 549.2, 590.6, 648.7, 703.3])
s2y = np.array([487.8, 443.8, 409.5, 359.3, 337.2, 340.8, 348.7, 372.5])
stx = np.array([304.2, 329.8, 387.0, 424.9, 455.7, 452.2, 412.6, 394.1])
sty = np.array([494.9, 468.5, 404.2, 385.7, 406.8, 438.5, 493.1, 508.1])


P1 = np.stack((p1x, p1y, np.ones(p1x.shape[0])), axis=-1).T
P2 = np.stack((p2x, p2y, np.ones(p2x.shape[0])), axis=-1).T

S1 = np.stack((s1x, s1y, np.ones(s1x.shape[0])), axis=-1).T
S2 = np.stack((s2x, s2y, np.ones(s2x.shape[0])), axis=-1).T

# fig, axs = plt.subplots(1, 3)
fig = plt.figure()
gs = GridSpec(2, 3, left=0.05, right=0.48)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1:3])
# axs[0].imshow(img1, cmap='gray')
# axs[0].scatter(P1[0, :], P1[1, :])
# axs[0].scatter(S1[0, :], S1[1, :])
# axs[1].imshow(img2, cmap='gray')
# axs[1].scatter(P2[0, :], P2[1, :])
# axs[1].scatter(S2[0, :], S2[1, :])
ax1.imshow(img1, cmap='gray')
ax1.scatter(P1[0, :], P1[1, :], s=5)
ax1.scatter(S1[0, :], S1[1, :], s=5)
ax2.imshow(img2, cmap='gray')
ax2.scatter(P2[0, :], P2[1, :], s=5)
ax2.scatter(S2[0, :], S2[1, :], s=5)

plt.xlim((0, 756))
plt.ylim((0, 756))
plt.gca().invert_yaxis()
ax3.set_box_aspect(1)
plot_line(ax3, P1, P2)
plot_line(ax3, S1, S2)

plt.show()