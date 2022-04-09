import numpy as np

def initialize_weights(self,l0,l1):
        
        '''
        l0 -- previous layer size
        l1 -- next layer size
        returns -- small random weights
        '''
        # He Initialization
        limit = np.sqrt(2 / float(l0))
        w = np.random.normal(0.0, limit, size=(l0,l1))
        b = np.zeros((1,l1))

        '''
        1. Random Initialization

        w = np.random.rand(l0,l1) * self.learning_rate
        b = np.zeros((1,l1))

        2. Xavier Initialization with Uniform Distribution
        
        limit = np.sqrt(6 / float(l0 + l1))
        w = np.random.uniform(low=-limit, high=limit, size=(l0, l1))

        3. LeCun Initilazation

        limit = np.sqrt(3 / float(l0))
        w = np.random.uniform(low=-limit, high=limit, size=(l0.l1))

        try other weight initialization techniques as well
        '''
        return w,b