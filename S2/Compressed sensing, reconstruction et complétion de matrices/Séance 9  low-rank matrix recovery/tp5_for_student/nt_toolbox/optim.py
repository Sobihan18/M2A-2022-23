import numpy as np
from nt_toolbox.compute_wavelet_filter import compute_wavelet_filter
from nt_toolbox.signal import perform_wavortho_transf

def MinL1_DR(y, A, niter=1000):
    """Solvers LA min using Douglas-Rachford."""
    n = A.shape[1]
    proxG = lambda x, gamma: np.fmax(0, 1 - gamma / np.fmax(1e-15, np.abs(x))) * x
    
    pA = A.conjugate().T.dot(np.linalg.inv( A.dot(A.conjugate().T) ))
    proxF = lambda x, y: x + pA.dot(y-A.dot(x))
    
    mu = 1
    gamma = 1
    
    rproxG = lambda x, tau: 2*proxG(x, tau) - x
    rproxF = lambda x, y: 2*proxF(x, y) - x
    
    lun = []
    err = []
    tx = np.zeros(n)
    for i in range(niter):
        tx = (1-mu/2) * tx + mu/2*rproxG(rproxF(tx, y), gamma)
        x = proxF(tx, y)
        lun.append(np.linalg.norm(x, ord=1))
        err.append(np.linalg.norm(y-A.dot(x)))
    return x.real, lun, err


def MinL1_fw_2d(scheme, measurements, level, hparams=("Daubechies",10), niter=50):
    """Solve the L1-problem:
        min ||x||_1 s.t. Ax=y when A is a partial Fourier-Wavelet matrix
        with Douglas-Rachford algorithm.
    """
    n = scheme.shape[0]
    #scheme_shifted = np.roll(np.roll(scheme, n/2, axis=0), n/2, axis=1)  # Remettre les basses frequences dans les coins
    scheme_shifted = np.fft.fftshift(scheme)
    J = np.log2(n)
    Jmin = max(1, J - level)
    h = compute_wavelet_filter(*hparams)
    
    multA = lambda x: scheme_shifted * np.fft.fft2(perform_wavortho_transf(x, Jmin, -1, h)) / n
    proxG = lambda x, tau: np.fmax(0, 1-tau/np.fmax(1e-15, np.abs(x))) * x
    proxF = lambda x, y: x + perform_wavortho_transf(n*np.fft.ifft2(y-multA(x)), Jmin, 1, h)
    
    mu = 1
    gamma = 1
    
    rproxG = lambda x, tau: 2 * proxG(x, tau) - x
    rproxF = lambda x, y: 2 * proxF(x, y) - x
    
    lun = []
    err = []
    tx = np.zeros((n, n))
    for i in range(niter):
        tx = (1-mu/2) * tx + mu/2*rproxG(rproxF(tx, measurements), gamma)
        x = proxF(tx, measurements)
        lun.append(np.linalg.norm(x, ord=1))
        err.append(np.linalg.norm(measurements-multA(x)))
    return x.real, lun, err



def MinL1_f_2d(scheme, measurements, niter=50):
    """Solve the L1-problem:
        min ||x||_1 s.t. Ax=y when A is a partial Fourier matrix
        with Douglas-Rachford algorithm.
    """
    n = scheme.shape[0]
    #scheme_shifted = np.roll(np.roll(scheme, n/2, axis=0), n/2, axis=1)  # Remettre les basses frequences dans les coins
    scheme_shifted = np.fft.fftshift(scheme)
    
    multA = lambda x: scheme_shifted * np.fft.fft2(x) / n
    proxG = lambda x, tau: np.fmax(0, 1-tau/np.fmax(1e-15, np.abs(x))) * x
    proxF = lambda x, y: x + n*np.fft.ifft2(y-multA(x))
    
    mu = 1
    gamma = 1
    
    rproxG = lambda x, tau: 2 * proxG(x, tau) - x
    rproxF = lambda x, y: 2 * proxF(x, y) - x
    
    lun = []
    err = []
    tx = np.zeros((n, n))
    for i in range(niter):
        tx = (1-mu/2) * tx + mu/2*rproxG(rproxF(tx, measurements), gamma)
        x = proxF(tx, measurements)
        lun.append(np.linalg.norm(x, ord=1))
        err.append(np.linalg.norm(measurements-multA(x)))
    return x.real, lun, err
    



def MinL1_f_1d(scheme, measurements, niter=50):
    """Solve the L1-problem:
        min ||x||_1 s.t. Ax=y when A is a partial Fourier matrix
        with Douglas-Rachford algorithm.
    """
    n = scheme.shape[0]
    #scheme_shifted = np.roll(np.roll(scheme, n/2, axis=0), n/2, axis=1)  # Remettre les basses frequences dans les coins
    scheme_shifted = np.fft.fftshift(scheme)
    
    multA = lambda x: scheme_shifted * np.fft.fft(x) / np.sqrt(n)
    proxG = lambda x, tau: np.fmax(0, 1-tau/np.fmax(1e-15, np.abs(x))) * x
    proxF = lambda x, y: x + np.sqrt(n)*np.fft.ifft(y-multA(x))
    
    mu = 1
    gamma = 1
    
    rproxG = lambda x, tau: 2 * proxG(x, tau) - x
    rproxF = lambda x, y: 2 * proxF(x, y) - x
    
    lun = []
    err = []
    tx = np.zeros(n)
    for i in range(niter):
        tx = (1-mu/2) * tx + mu/2*rproxG(rproxF(tx, measurements), gamma)
        x = proxF(tx, measurements)
        lun.append(np.linalg.norm(x, ord=1))
        err.append(np.linalg.norm(measurements-multA(x)))
    return x.real, lun, err 