import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.preprocessing import KernelCenterer
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from mpl_toolkits.mplot3d import Axes3D


def generate_data():
    np.random.seed(0)
    X, y = make_circles(n_samples=400, factor=.3, noise=.05)
    return X,y

def rbfkernel(gamma, distance):
    return np.exp(-gamma * distance)

def squared_euclidean_distance(data):
    sq_dists = pdist(data, 'sqeuclidean')
    return sq_dists

def KPCA(gamma, data, feature_size):
    sq_dists = squared_euclidean_distance(data)
    # squareform to converts the pairwise distances into a symmetric 400x400 matrix
    mat_sq_dists = squareform(sq_dists)
    # Compute the 400x400 kernel matrix
    K = rbfkernel(gamma, mat_sq_dists)
    # Center the kernel matrix
    kern_cent = KernelCenterer()
    K = kern_cent.fit_transform(K)
    # Get the eigenvector with largest eigenvalue
    eigen_values, eigen_vectors = eigh(K)
    indexes = eigen_values.argsort()[::-1]
    direction_vectors = eigen_vectors[:, indexes[0: feature_size]]
    projected_data = np.dot(K, direction_vectors)

    return projected_data

def plot3D_KPCA(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a = ax.scatter([x[i][0] for i in range(len(x)) if y[i] == 0], [x[i][1] for i in range(len(x)) if y[i] == 0], [x[i][2] for i in range(len(x)) if y[i] == 0], color='red')
    b = ax.scatter([x[i][0] for i in range(len(x)) if y[i] == 1], [x[i][1] for i in range(len(x)) if y[i] == 1], [x[i][2] for i in range(len(x)) if y[i] == 1], color='blue')
    ax.legend([a, b],['Class 0','Class 1'])
    plt.title('First component after RBF Kernel PCA in 3D')
    plt.show()

def plot2D_KPCA(x,y):
    plt.figure(figsize=(8,8))
    plt.scatter([x[i][0] for i in range(len(x)) if y[i] == 0], [x[i][1] for i in range(len(x)) if y[i] == 0], color='red')
    plt.scatter([x[i][0] for i in range(len(x)) if y[i] == 1], [x[i][1] for i in range(len(x)) if y[i] == 1], color='blue')
    plt.title('First component after RBF Kernel PCA in 2D')
    plt.show()

def plot_data(x, y):
	plt.figure(figsize=(6,6))
	a = plt.scatter(x[y==0, 0], x[y==0, 1], color='red')
	b = plt.scatter(x[y==1, 0], x[y==1, 1], color='blue')
	plt.title('A nonlinear 2Ddataset')
	plt.ylabel('y coordinate')
	plt.xlabel('x coordinate')
	plt.legend([a, b],['Class 0','Class 1'])
	plt.show()

def main():
	# gamma can be take arbitarilay any integer between 2-10
    gamma=2
    # Data is generated where x is the "x-y coordinate of data points" and y is the label namely 0 and 1
    x,y = generate_data()
    # Plotting the Actual Dataset
    plot_data(x,y)
    # Calling the Kernal Principal Component Analysis Function with feature_size=2
    Kpca_2D = KPCA(gamma,x,2)
    # Plotting the resulting 3D graph
    plot2D_KPCA(Kpca_2D,y)
    # Calling the Kernal Principal Component Analysis Function with feature_size=2
    Kpca_3D = KPCA(gamma,x,3)
    # Plotting the resulting 3D graph
    plot3D_KPCA(Kpca_3D,y)

if __name__ == '__main__':
	main()
