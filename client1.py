import os
import flwr as fl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils
import seaborn as sns

# Make TensorFlow log less verbose
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ['GRPC_TRACE'] = 'all'
# os.environ['GRPC_VERBOSITY'] = 'DEBUG'


def getDist(y):
    ax = sns.countplot(y)
    ax.set(title="Count of Client1 data classes")
    plt.show()


def getData(dist, x, y):
    # print(len(x), len(y))
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]] < dist[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1
    return np.array(dx), np.array(dy)


# Load model and data
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, input_shape=(1, 28, 28), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 1, 28, 28)/255
x_test = x_test.reshape(10000, 1, 28, 28)/255

dist = [4000, 4000, 4000, 3000, 10, 10, 10, 10, 4000, 10]
x_train, y_train = getData(dist, x_train, y_train)
getDist(y_train)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters) # update the parameters from the server
        r = model.fit(x_train, y_train, epochs=1, batch_size=32)
        # history = r.history
        # print('Fit history:', history)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        # print('Eval accuracy:', accuracy)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    "localhost:7001",
    client=CifarClient(),
    # grpc_max_message_length=1024 * 1024 * 1024
)
