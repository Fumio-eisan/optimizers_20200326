import numpy as np

class SGD:

    def __init__(self, lr = 0.01,x=100):
        self.lr = 0.01
        self.x =100.0

    def update(self,x):
        x+= -self.lr * 2*x 
        return x


class Momentum:

    def __init__(self,lr=0.01,alpha=0.8,v=None):
        self.lr = 0.01
        self.v = 0.0
        self.alpha = 0.8

    def update(self,x):
        self.v = self.v*self.alpha  - (2.0)*x*self.lr 
        x += + self.v
        return x

    
class AdaGrad:

    def __init__(self,h0=10.0):
        self.v = 0.0
        self.h = 0.0
        self.h0 = 10.0

    def update(self,x):
        self.h += + (2.0*x)**2
        self.v = -self.h0/(np.sqrt(self.h)+1e-7)
        x += + self.v*2*x
        return x
    