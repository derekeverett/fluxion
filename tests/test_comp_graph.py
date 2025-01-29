import unittest
import numpy as np
import gradflow.comp_graph as cg
import autograd

np.random.seed(42)


class TestCompGraph(unittest.TestCase):

    def setUp(self):
        d_in = 7 # dimensionality of input vectors
        d_out = 1 # dimensionality of output vectors
        d_hid = 13 # dimensionality of vectors for hidden layer
        B = 17 # batch size

        # initial weights for the hidden layers
        self.w_A_np = np.random.normal(size=(d_in, d_hid))
        self.w_C_np = np.random.normal(size=(d_hid, d_out))

        # inputs
        self.x_np = np.random.normal(size=(B, d_in))
        # weights in the true data generating function
        omega_true = np.random.normal(size=(d_in, 1))
        # data generating function
        def true_fct(z):
            return np.cos(np.dot(z, omega_true))
        # true outputs
        self.y_true = np.array([true_fct(z) for z in self.x_np])

    def ag_test_function(self, input):
        z = autograd.numpy.dot(input, self.w_A_np)
        z = autograd.numpy.tanh(z)
        z = autograd.numpy.dot(z, self.w_C_np)
        l = autograd.numpy.mean( (z - self.y_true)**2. )
        return l, z
        
    def test_NN(self):

        # define the inputs, parameters, and network function nodes
        x = cg.Value("x", self.x_np) # input vector
        w_A = cg.Value("w_A", self.w_A_np) # params in first linear layer
        linear1 = cg.Dot("linear1") # first linear layer
        act1 = cg.Tanh("act1") # activation function
        w_C = cg.Value("w_C", self.w_C_np) # params in second linear layer
        linear2 = cg.Dot("linear2") # second linear layer
        loss = cg.MSELoss("mse") # mean squared error loss
        
        # .forward() calls create the function topology
        x.forward()
        w_A.forward()
        w_C.forward()
        y = linear1.forward(x, w_A)
        y = act1.forward(linear1)
        y = linear2.forward(act1, w_C)
        l = loss.forward(linear2, self.y_true)

        l_ag, y_ag = self.ag_test_function(self.x_np)
        
        result   = y
        expected = y_ag
        assert np.allclose(result, expected), f"CNOT test failed: {result} != {expected}"

        result   = l
        expected = l_ag
        assert np.allclose(result, expected), f"CNOT test failed: {result} != {expected}"


if __name__ == '__main__':
    unittest.main()