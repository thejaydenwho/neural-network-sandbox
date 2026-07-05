import tensorflow as tf
from Base import *
from Activations import *
from Layers import *
from Loss import *
from Optimizers import *

# Import the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print(X_train.shape)
print(y_train.shape)
# Load into training and testing sets

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255.0
X_test /= 255.0 

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

number_training = Sequential([Dense(784,256), ReLU(), Dense(256, 128), ReLU(), Dense(128, 10), Softmax()])

Benchmarker = Trainer(number_training, CrossEntropy(), MomentumSGD(0.01, 0.9) )
Benchmarker.test(X_test, y_test)
Benchmarker.train(X_train, y_train, 100, 32)
Benchmarker.test(X_test, y_test)