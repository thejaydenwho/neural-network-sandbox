import tensorflow as tf
from Base import *
from Activations import *
from Layers import *
from Loss import *
from Optimizers import *
import time

# Import the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print(X_train.shape)
print(y_train.shape)
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255.0
X_test /= 255.0 

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

fashion_training = Sequential([Conv2D(in_channels=1,out_channels=16,kernel_size=(3,3))])
'''
number_training = Sequential([Dense(784,256), ReLU(), Dense(256, 128), ReLU(), Dense(128, 10), Softmax()])

Benchmarker = Trainer(number_training, CrossEntropy(), MomentumSGD(0.01, 0.9) )
Benchmarker.test(X_test, y_test)
Benchmarker.train(X_train, y_train, 100, 32)
Benchmarker.test(X_test, y_test)
'''
model = Sequential([
    Conv2D(1, 16, (3,3), padding=(1,1)),
    ReLU(),

    Conv2D(16, 16, (3,3), padding=(1,1)),
    ReLU(),

    MaxPool2D(),

    Conv2D(16, 32, (3,3), padding=(1,1)),
    ReLU(),

    MaxPool2D(),

    Flatten(),

    Dropout(0.3),

    Dense(7*7*32, 128),
    ReLU(),

    Dropout(0.3),

    Dense(128, 10),
    Softmax()
])

Benchmarker1 = Trainer(model, CrossEntropy(), Adam())
Benchmarker1.test(X_test, y_test)
start_time = time.perf_counter()
Benchmarker1.train(X_train, y_train, epochs=30, batch_size=64, csv_file="Adam_Training.csv")
end_time = time.perf_counter()
print(end_time - start_time)
Benchmarker1.test(X_test, y_test)

Benchmarker2 = Trainer(model, CrossEntropy(), MomentumSGD())
Benchmarker2.test(X_test, y_test)
start_time = time.perf_counter()
Benchmarker2.train(X_train, y_train, epochs=30, batch_size=64, csv_file="MomentumSGD_Training.csv")
end_time = time.perf_counter()
print(end_time - start_time)
Benchmarker2.test(X_test, y_test)
