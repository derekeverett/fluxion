import unittest
import numpy as np
import fluxion.comp_graph as cg

np.random.seed(42)


class TestOpt(unittest.TestCase):

    def setUp(self):
        self.epochs = 1000  # number of epochs
        self.lr = 1e-2  # learning rate
        d_in = 1  # dimensionality of input vectors

        # initialize inputs
        self.x_np = np.random.normal(size=(1, d_in))

        # solution to min_x f(x) = x^2
        self.soln = np.zeros_like(self.x_np)

    def test_grad_desc(self):

        # define the inputs, parameters, and network function nodes
        x = cg.Value("x", self.x_np, optimize=True)  # input vector
        loss = cg.MSELoss("mse")  # mean squared error loss

        for epoch in range(self.epochs):
            x.forward()
            _ = loss.forward(x, self.soln)
            # call backward pass on topo sorted graph
            fn_graph = cg.Graph("f(x)", [loss])
            fn_graph.backward()
            # step the optimizer to update weights
            fn_graph.step_optimizer(lr=self.lr)

        # checks the embeddings
        result = x.out
        expected = self.soln
        assert np.allclose(
            result, expected, atol=1e-6
        ), f"Gradient descent on f(x) = x^2 did not converge within tolerance: {result} != {expected}"


if __name__ == "__main__":
    unittest.main()
