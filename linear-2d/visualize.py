import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

data = np.genfromtxt('length_weight.csv', delimiter=',')
x_train = np.transpose(np.mat(data[:, 0]))
y_train = np.transpose(np.mat(data[:, 1]))

ax.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax.set_xlabel('x (length)')
ax.set_ylabel('y (weight)')


class LinearRegressionModel:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return x * self.W + self.b

    # Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))


model = LinearRegressionModel(np.mat([[0.23805696]]), np.mat([[-8.483095]]))

x = np.mat([[np.min(x_train)], [np.max(x_train)]])
ax.plot(x, model.f(x), label='$y = f(x) = xW+b$')

print('loss:', model.loss(x_train, y_train))

ax.legend()
plt.show()
