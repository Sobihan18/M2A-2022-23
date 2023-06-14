#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:57:34 2017

@author: maxime
"""

import getpass
import numpy as np
import matplotlib.pyplot as plt
from nt_toolbox.general import rescale
from nt_toolbox.signal import load_image, perform_wavortho_transf
from nt_toolbox.compute_wavelet_filter import compute_wavelet_filter

def get_measurements(n=32, r_sparse=0.2, r_info=0.5):
    """
    Measurement function.
    
    Parameters:
    - n is the image size (n x n);
    - r_sparse is the ratio of non-zero coefficients (wavelet domain) of the
    signal x to recover;
    - r_info is the ratio between the size of y and the size of x.
    
    Return y, A, T, where:
    - y is the vector of measurements;
    - A is the sensing matrix (we look for x such that y = Ax);
    - T is a total variation operator.
    """
    np.random.seed(sum([ord(c) for c in getpass.getuser()]))
    i = np.random.randint(0, 11)
    
    im = rescale(load_image("data/" + str(i) + ".bmp", n))
    
    h = compute_wavelet_filter("Daubechies", 4)

    # Compute the matrix of wavelet transform
    mask = np.zeros((n, n))
    A0 = []
    for i in range(n):
        for j in range(n):
            mask[i, j] = 1
            wt = perform_wavortho_transf(mask, 0, +1, h)
            A0.append(wt.ravel())
            mask[i, j] = 0
    A0 = np.asarray(A0)
    
    # Gaussian matrix x Wavelet transform (keep ratio r_info)
    G = np.random.randn(int(np.floor(n**2*r_info)), n**2) / n
    A = G.dot(A0)

    # Threshold the image (keep ratio r_sparse) and generate the measurements y
    x_true = perform_wavortho_transf(im, 0, +1, h).ravel()  # Same as x_true = A0.T.dot(im.flatten())
    thshol = np.sort(np.abs(x_true.ravel()))[int((1-r_sparse)*n**2)]
    x_true[np.abs(x_true) <= thshol] = 0
    y = A.dot(x_true)  # Vector of measurements
    
    # Total variation operator
    dx = np.eye(n**2)
    dx -= np.roll(dx, 1, axis=1)
    dx = np.delete(dx, np.s_[n-1::n], axis=0)

    dy = np.eye(n**2)
    dy -= np.roll(dy, n, axis=1)
    dy = np.delete(dy, np.s_[-n:], axis=0)

    T = np.r_[dx, dy].dot(A0)  # TV in the image domain

    T = np.r_[np.eye(n**2), T]  # For usual L1 norm, add identity
    
    return y, A, T

def back_to_image(x):
    n = int(np.sqrt(x.size))
    h = compute_wavelet_filter("Daubechies", 4)
    wt = x.reshape((n, n))
    im = perform_wavortho_transf(wt, 0, -1, h)
    return im

def plot_image(x):
    plt.figure(figsize=(1, 1))
    im = back_to_image(x)
    plt.imshow(im, cmap='gray')
    plt.axis('off')

def minL2(y, A):
    p, n = A.shape
    if p < n:
        return A.T.dot( np.linalg.solve(A.dot(A.T), y) )
    else:
        return np.linalg.solve(A.T.dot(A), A.T.dot(y))
