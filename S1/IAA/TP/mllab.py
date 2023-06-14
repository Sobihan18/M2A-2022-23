"""
Packages:
	nympy as np
	matplotlib.pyplot as plt
	seaborn as sns

Functions:
	plotXY
	plot_frontiere
	map_regions
	covariance
	plot_cov
	sample_gmm
	scatter
	plot_level_set
	gaussian_sample
"""

print(__doc__)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
sns.set()


def plotXY(X, Y, legend=True):
    """
        Scatter points with a color for each class.
        Input:
            X and Y may be:
            - two numpy arrays with two columns; each array is the data matrix for a class (works only for
            two classes).
            - a numpy array with two columns (the data matrix) and the vector of labels (works for many classes).
    """    
    if Y.ndim > 1:
        X1 = X
        X2 = Y
        XX = np.concatenate((X, Y), axis=0)
        YY = np.concatenate((np.ones(X.shape[0]), -np.ones(Y.shape[0])))
    else:
        XX = X
        YY = Y
    for icl, cl in enumerate(np.unique(YY)):
        plt.scatter(XX[YY==cl, 0], XX[YY==cl, 1], label='Class {0:d}'.format(icl+1))
    plt.axis('equal')
    if legend:
        plt.legend()


# def plot_frontiere(clf, data=None, num=500, label=None):
#     """
#         Plot the frontiere f(x)=0 of the classifier clf within the same range as the one
#         of the data.
#         Input:
#             clf: binary classifier with a method decision_function
#             data: input data (X)
#             num: discretization parameter
#     """
#     xmin, ymin = data.min(axis=0)
#     xmax, ymax = data.max(axis=0)
#     x, y = np.meshgrid(np.linspace(xmin, xmax, num), np.linspace(ymin, ymax))
#     z = clf.decision_function(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
#     cs = plt.contour(x, y, z, [0], colors='r')
#     if label is not None:
#         cs.levels = [label]
#         plt.gca().clabel(cs)
#     return cs


# def map_regions(clf, data=None, num=500):
#     """
#         Map the regions f(x)=1…K of the classifier clf within the same range as the one
#         of the data.
#         Input:
#             clf: classifier with a method predict
#             data: input data (X)
#             num: discretization parameter
#     """
#     xmin, ymin = data.min(axis=0)
#     xmax, ymax = data.max(axis=0)
#     x, y = np.meshgrid(np.linspace(xmin, xmax, num), np.linspace(ymin, ymax))
#     z = clf.predict(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
#     zmin, zmax = z.min(), z.max()
#     plt.imshow(z, origin='lower', interpolation="nearest",
#                extent=[xmin, xmax, ymin, ymax], cmap=cm.coolwarm,
#               alpha=0.3)


# def plot_frontiere(functions, data=None, data_labels=None, num=500, label=None):
#     """
#         Plot the frontiere fun(x)=0 of the classifier clf within the same range as the one
#         of the data.
#         Input:
#             functions: scoring function or list of scoring functions (typically clf.decision_function)
#             data: input data (X)
#             data_labels: data labels (y)
#             num: discretization parameter
#             label: scoring functions labels as a list
#     """
#     if not hasattr(functions, '__iter__'):
#         functions = [functions]
#     if label is not None and not hasattr(label, '__iter__'):
#         label = [label]
        
#     xmin, ymin = data.min(axis=0)
#     xmax, ymax = data.max(axis=0)
#     x, y = np.meshgrid(np.linspace(xmin, xmax, num), np.linspace(ymin, ymax))
    
#     plt.figure(figsize=(7, 7))
#     plt.scatter(*data.T, c=data_labels, cmap='plasma')
    
#     for i, fun in enumerate(functions):
#         z = fun(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
#         cs = plt.contour(x, y, z, [0], colors='r')
#         if label is not None:
#             cs.levels = [label[i]]
#             plt.gca().clabel(cs)
#     plt.axis('image')
#     minx, miny = data[:, 0].min(), data[:, 1].min()
#     diffx, diffy = data[:, 0].max() - minx, data[:, 1].max() - miny
#     plt.axis([minx - 0.1*diffx, minx + 1.1*diffx, miny - 0.1*diffy, miny + 1.1*diffy])


def plot_frontiere(clfs, data=None, data_labels=None, label=None, num=500, figure=True):
    """
        Plot the frontiere fun(x)=0 of the classifier clf within the same range as the one
        of the data.
        Input:
            clfs: classifier or list of classifiers
            data: input data (X)
            data_labels: data labels (y)
            label: classifier labels as a list
            num: discretization parameter
            figure: create a new figure
    """
    if not hasattr(clfs, '__iter__'):
        clfs = [clfs]
    if label is not None and not hasattr(label, '__iter__'):
        label = [label]
        
    xmin, ymin = data.min(axis=0)
    xmax, ymax = data.max(axis=0)
    x, y = np.meshgrid(np.linspace(xmin, xmax, num), np.linspace(ymin, ymax))
    
    if figure:
        plt.figure(figsize=(7, 7))
#     plt.scatter(*data.T, c=data_labels, cmap='plasma')
    for icl, cl in enumerate(np.unique(data_labels)):
        plt.scatter(*data[data_labels==cl].T, label='Class {0:d}'.format(icl+1))
        
    for i, clf in enumerate(clfs):
        z = clf.decision_function(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
        cs = plt.contour(x, y, z, [0], colors='r')
        if label is not None:
            cs.levels = [label[i]]
            plt.gca().clabel(cs)
    if figure:
        plt.axis('image')
    minx, miny = data[:, 0].min(), data[:, 1].min()
    diffx, diffy = data[:, 0].max() - minx, data[:, 1].max() - miny
    plt.axis([minx - 0.1*diffx, minx + 1.1*diffx, miny - 0.1*diffy, miny + 1.1*diffy])
    plt.legend(loc="best")
    
    
def map_regions(clf, data=None, data_labels=None, num=200):
    """
        Map the regions f(x)=1…K of the classifier clf within the same range as the one
        of the data.
        Input:
            clf: classifier with a method predict
            data: input data (X)
            data_labels: data labels (y)
            num: discretization parameter
    """
    xmin, ymin = data.min(axis=0)
    xmax, ymax = data.max(axis=0)
    x, y = np.meshgrid(np.linspace(xmin, xmax, num), np.linspace(ymin, ymax))
    z = clf.predict(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
    zmin, zmax = z.min(), z.max()
    for icl, cl in enumerate(np.unique(data_labels)):
        plt.scatter(*data[data_labels==cl].T, label='Class {0:d}'.format(icl+1))
    plt.imshow(z, origin='lower', interpolation="nearest",
               extent=[xmin, xmax, ymin, ymax], cmap=cm.coolwarm,
              alpha=0.3)
    plt.axis('equal')
    minx, miny = data[:, 0].min(), data[:, 1].min()
    diffx, diffy = data[:, 0].max() - minx, data[:, 1].max() - miny
    plt.axis([minx - 0.1*diffx, minx + 1.1*diffx, miny - 0.1*diffy, miny + 1.1*diffy])
    plt.legend()


def covariance(sigma1=1., sigma2=1., theta=0.):
    """
        Covariance matrix with eigenvalues sigma1 and sigma2, rotated by the angle theta.
    """
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    cov = np.array([[sigma1, 0.],
                   [0, sigma2]])
    return rotation.dot(cov.dot(rotation.T))


def plot_cov(cov, mean=[0, 0], cst=6, num=200):
    """Display the ellipse associated to the covariance matrix cov.
    If mean is specified, the ellipse is translated accordingly.
    """
    cov = np.linalg.inv(np.asarray(cov))
    mean = np.asarray(mean)
    theta = np.linspace(0, 2*np.pi, num=num)
    X = np.c_[np.cos(theta), np.sin(theta)]
    X = X.T * np.sqrt(cst / np.diag(X.dot(cov.dot(X.T))))
    X = X.T + mean
    plt.plot(X[:, 0], X[:, 1], 'r')
    

def sample_gm(weights, means, covariances, size=1):
    """Sample points from a Gaussian mixture model specified by the weights, the means
    and the covariances. These three parameters are lists."""
    X = None
    p = np.random.multinomial(1, weights, size=size)
    for (m, c, i) in zip(means, covariances, p.T):
        Y = np.random.multivariate_normal(m, c, size=size)
        if X is None:
            X = Y.copy()
        else:
            X[i==1] = Y[i==1]
    return X


def scatter(X, labels=None):
    """Scatter points with different colors.
    Input:
    - X: 2d-array like of size n x 2 (n is the number of points)
    - labels: list of point labels (of size n)
    """
    plt.scatter(X[:, 0], X[:, 1], c=np.arange(X.shape[0]))
    if labels is not None:
        for x, label in zip(X, labels):
            plt.annotate(label, xy=x,
                         xytext=x + np.r_[0.02, 0.02]*(X.max(axis=0)-X.min(axis=0)))


# def plot_level_set(red, data=None, num=500, label=None, num_levels=10, component=0):
#     """
#         Plot the level sets of the dimensionality reduction technique red within the same range as the one
#         of the data.
#         Input:
#             red: reduction technique with a method transform
#             data: input data (X)
#             num: discretization parameter
#             num_levels: number of level sets
#             component: reduced dimension of interest
#     """
#     xmin, ymin = data.min(axis=0)
#     xmax, ymax = data.max(axis=0)
#     x, y = np.meshgrid(np.linspace(xmin, xmax, num), np.linspace(ymin, ymax))
#     u = red.transform(np.c_[x.ravel(), y.ravel()])[:, component].reshape(x.shape)
#     for t in np.linspace(u.min(), u.max(), num_levels):
#         z = np.fabs(u+t)
#         zmin, zmax = z.min(), z.max()
#         ind = np.where((z-zmin)/(zmax-zmin) < 0.001)
#         ind_sort = np.argsort(y[ind])
#         plt.plot(x[ind][ind_sort], y[ind][ind_sort], '.', label=label, linewidth=2, color='black')
        
        
def plot_level_set(red, data=None, num=500, num_levels=10, component=0):
    """
        Plot the level sets of the dimensionality reduction technique red within the same range as the one
        of the data.
        Input:
            red: reduction technique with a method transform
            data: input data (X)
            num: discretization parameter
            num_levels: number of level sets
            component: reduced dimension of interest
    """
    xmin, ymin = data.min(axis=0)
    xmax, ymax = data.max(axis=0)
    x, y = np.meshgrid(np.linspace(xmin, xmax, num), np.linspace(ymin, ymax))
    u = red.transform(np.c_[x.ravel(), y.ravel()])[:, component].reshape(x.shape)
    cs = plt.contour(x, y, u, np.linspace(u.min(), u.max(), num_levels))
    cs.levels = np.linspace(u.min(), u.max(), num_levels)
    plt.gca().clabel(cs)


def gaussian_sample(mu=[0, 0], sigma1=1., sigma2=1., theta=0., n=50):
    cov = covariance(sigma1, sigma2, theta)
    x = multivariate_normal.rvs(mean=mu, cov=cov, size=n)
    return x



