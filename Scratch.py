from Base import *
from Activations import *
from Layers import *
from Loss import *
from Optimizers import *
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255.0
X_test /= 255.0 

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]


number_training = Sequential([Dense(784,256), ReLU(), Dense(256, 256), ReLU(), Dense(256, 10), Softmax()])

Benchmarker = Trainer(number_training, CrossEntropy(), MomentumSGD(0.01, 0.9) )
Benchmarker.test(X_test, y_test)
Benchmarker.train(X_train, y_train, 10, 32)
Benchmarker.test(X_test, y_test)
'''
seq = Sequential([Dense(2,32), ReLU(), Dense(32,1), Sigmoid()])


np.random.seed(0)

X = np.random.randn(1000, 2)

# circle: x^2 + y^2 < 1
y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)
y = y.reshape(-1, 1)
print(X)
print(y)

Benchmarker = Trainer(seq, MSE(), MomentumSGD(0.01, 0.9))

Benchmarker.train(X,y,10000,32)

X = np.random.randn(1000, 2)

# circle: x^2 + y^2 < 1
y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)
y = y.reshape(-1, 1)

Benchmarker.test(X,y)
'''
