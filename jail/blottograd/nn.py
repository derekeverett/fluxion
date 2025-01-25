import numpy as np

class Identity():
    def __init__(self):
        pass

    def forward_fct(self, input, weights):
        return input
    
    def vjp_fct(self, input, weights, vec):
        return vec

class Linear():
    def __init__(self):
        pass
    
    def forward_fct(self, input, weights):
        return np.dot(weights, input)

    def vjp_fct(self, input, weights, vec):
        return np.dot(vec, weights)

class Tanh():
    def __init__(self):
        pass

    def forward_fct(self, input, weights):
        return np.tanh(input)
    
    def vjp_fct(self, input, weights, vec):
        sech2 = 1./ (np.cosh(input)**2.)
        return vec * sech2