import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

data = np.genfromtxt('day_head_circumference.csv', delimiter=',')
x_train = np.transpose(np.mat(data[:, 0]))
y_train = np.transpose(np.mat(data[:, 1]))

ax.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax.set_xlabel('x (day)')
ax.set_ylabel('y (head circumference)')


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


class NonlinearRegressionModel:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return 20 * sigmoid(x * self.W + self.b) + 31

    # Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))


model = NonlinearRegressionModel(np.mat([[0.00259391]]), np.mat([[-0.00122474]]))

x = np.linspace(np.min(x_train), np.max(x_train)).reshape(-1, 1)
ax.plot(x, model.f(x), label='$y = f(x) = 20\\sigma(xW+b) + 31$')

print('loss:', model.loss(x_train, y_train))

ax.legend()
plt.show()
