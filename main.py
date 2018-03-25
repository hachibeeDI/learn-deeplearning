import numpy as np
from matplotlib import pylab as plt

from src import exp


class Chap_3_3:
    def confirm_sigmoid():
        x = np.array([-1.0, 1.0, 2.0])
        y = exp.sigmoid(x)
        plt.plot(x,y)
        plt.ylim(-0.1, 1.1)
        plt.show()

    def confirm_exp_3_9():
        """
        p60-p61
        A weighted addition for the first neuron of first layer.

        W = weight?
        B = bias
        Z = zone?

        b + w1x1 + w2x2 => a1
        """
        X = np.array([1.0, 0.5])
        W1 = np.array([[1.0, 0.3, 0.5], [0.2, 0.4, 0.6]])
        B1 = np.array([0.1, 0.2, 0.3])

        A1 = np.dot(X, W1) + B1
        return A1

    def confirm_activate_function_applied():
        """
        p62

        h(a1) => z1
        """
        A1 = self.confirm_exp_3_9()
        Z1 = exp.sigmoid(A1)
        print(A1)
        print(Z1)


def init_network():
    return {
        'W1': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
        'b1': np.array([0.1, 0.2, 0.3]),

        'W2': np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
        'b2': np.array([0.1, 0.2]),

        'W3': np.array([[0.1, 0.3], [0.2, 0.4]]),
        'b3': np.array([0.1, 0.2]),
    }


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = exp.sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = exp.sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = exp.identity(a3)
    return y


if __name__ == '__main__':
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)
    assert y[0] == 0.3168270764110298, y[0]
    assert y[1] == 0.6962790898619668, y[1]
