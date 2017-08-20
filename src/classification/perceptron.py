import numpy as np

class Perceptron():
    """Perceptron classifier.

    Artificial neuron, auto learns optimal weights. Thresholded perceptron model. All-or-nothing firing.
    If input > threshold, activate, classified as positive label.

    For each Sample:
    x (feature) ---> w (feature weights) ---> Net Input (z) ---> Activation A(z) ---> Output (y)
                        |                                                          |
                        |______________________Wj (weight update)__________________|

    Learning rule:
        Wj = learning rate * (Y - y) * x
    Weight adjustment rule, until Y output = Y label expected.

    Activation: unit step function
        A(z): YES label = if z > thres, z > 0 if centered
               NO label = otherwise
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.learn_rate = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fitting training data. Model creation.

        Pass in X samples, row = sample, col = feature. And Y labels. Expected class labels
        """
        self._weights = np.zeros(1 + X.shape[1])    # pad one extra w0 = -w, zero centering of thres for simplicity
        self._errors = []                           # tally num of misclassifications [1,0,1,1,0,1,0...]

        for _ in range(self.n_iter):
            error = 0

            for xi, target in zip(X, y):
                update = self.learn_rate * (target - self.predict(xi))
                self._weights[0] += update          # w0
                self._weights[1:] += update * xi    # vector update
                error += int(update != 0.0)         # error rate, if condition true = 1, false = 0

            self._errors.append(error)

        return self

    def predict(self, X):
        """Predicts class label. After unit step

        Z > 0, label YES, otherwise NO. 0 = threshold, since zero centered
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        """Calc net input. Z = wT * x  vector dot prod
        """
        return np.dot(self._weights[1:], X) + self._weights[0]  # center Z thres around 0
