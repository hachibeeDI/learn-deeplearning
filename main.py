from matplotlib import pylab as plt

from .src import exp


class Chap_3_3
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
        """
        X = np.array([1.0, 0.5])
        W1 = np.array([[1.0, 0.3, 0.5], [0.2, 0.4, 0.6]])
        # bias of the first layer
        B1 = np.array([0.1, 0.2, 0.3])

        A1 = np.dot(X, W1) + B1
        return A1

    def confirm_activate_function_applied():
        """
        p62
        """
        A1 = self.confirm_exp_3_9()
        Z1 = exp.sigmoid(A1)
        print(A1)
        print(Z1)
