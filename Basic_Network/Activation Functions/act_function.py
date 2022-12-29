import numpy as np

def forward_prop(alpha = 0.01):

    # Leaky-Relu
    a1 = np.maximum(alpha*z1, z1)

    '''
    1. Relu Function   Max(0,x) x -- input
        a1 = np.maximum(0,z1)   

    2. Tan-h    (e ^ x - e ^-x )/ (e ^ x + e ^-x) x --input       

        a1 = np.tanh(z1)
        a1 = np.sinh(z1)/np.cosh(z1)

    '''