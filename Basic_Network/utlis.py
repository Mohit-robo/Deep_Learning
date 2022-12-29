import cmath
from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from pandas import array
from pyparsing import alphas


def create_dataset(N,K=2):

    # N = 100  # No of Data Points/ class
    D = 2
    X = np.zeros((N*K,D))  # data matix (each row = single example)
    y = np.zeros(N*K)  

    for j in range(K):
        ix = range(N*j,N * (j+1))
        r = np.linspace (0,1,N) # radius
        
        """
        Return evenly spaced numbers over a specified interval.

        Returns `N` evenly spaced samples, calculated over the
        interval [`0`, `1`]. In our case make 100 equal intervals of 0-1.
        """ 
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r*np.sin(t),r * np.cos(t)]
        y[ix] = j

        # visualizing the dataset:
        # h = 0.02
        
        # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min, y_max, h))
        # points = np.c_[xx.ravel(),yy.ravel()]
        # print(len(points))

        plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral)
        plt.show()
    
    return X,y  

def plot_contour(X,y,model,parameters):

    h = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min, y_max, h))

    points = np.c_[xx.ravel(),yy.ravel()]

    # forward prop with our trained parameters
    _,Z = model.forward_prop(points, parameters)

    # classify into highest prob
    Z = np.argmax(Z,axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,cmap = plt.cm.Spectral,alpha = 0.8)

    # plt the points
    plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral)
    plt.show()
