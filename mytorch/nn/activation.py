# Do not import any additional 3rd party external libraries as they will not
# be available to Gradescope and are not needed (or allowed)
import numpy as np

class Sigmoid:
    """
    Sigmoid activation function
    """
    def forward(self, Z):

        self.A = Z
        self.npVal = np.exp(-self.A)
        return 1 / (1 + self.npVal)

    def backward(self, dLdA):

        dAdZ = self.npVal / (1 + self.npVal) ** 2
        return dAdZ*dLdA

class Tanh:
    """
    Modified Tanh to work with BPTT.
    The tanh(x) result has to be stored elsewhere otherwise we will
    have to store results for multiple timesteps in this class for each cell,
    which could be considered bad design.

    Now in the derivative case, we can pass in the stored hidden state and
    compute the derivative for that state instead of the "current" stored state
    which could be anything.
    """
    def forward(self, Z):

        self.A = Z
        self.tanhVal =  np.tanh(self.A)
        return self.tanhVal

    def backward(self, dLdA, state=None):
        if state is not None:
            dAdZ = 1 - state*state
            return dAdZ * dLdA
        else:
            dAdZ = 1 - self.tanhVal * self.tanhVal
            return dAdZ * dLdA


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        Z_shifted = Z - np.max(Z, axis = self.dim, keepdims = True)
        exp_Z = np.exp(Z_shifted)
        self.A = exp_Z / np.sum(exp_Z, axis = self.dim, keepdims = True)
        return self.A


    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        dot = np.sum(dLdA * self.A, axis = self.dim, keepdims = True)
        dLdZ = self.A * (dLdA - dot)

        return dLdZ
 
