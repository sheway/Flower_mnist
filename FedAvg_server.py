from typing import Callable
from typing import Dict
import flwr as fl
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
import numpy as np
import os

# os.environ['GRPC_TRACE'] = 'all'
# os.environ['GRPC_VERBOSITY'] = 'DEBUG'


#Configuring client fit and client evaluate
def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    def fit_config(rnd: int) -> Dict[str, str]:
        config = {
            "learning_rate": str(0.001),
            "batch_size": 32,
        }
        return config
    return fit_config


if __name__ == "__main__":
    # model = Sequential()
    # model.add(Conv2D(filters=32, kernel_size=3, input_shape=(1, 28, 28), activation='relu', padding='same'))
    # model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(10, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        min_available_clients=2,  #最少要幾個 client 才能執行
        # initial_parameters=weights_to_parameters(model.get_weights()),
       on_fit_config_fn=get_on_fit_config_fn(),
    )

    # Start Flower server
    fl.server.start_server(
        server_address='localhost:7001',
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        # grpc_max_message_length=1024 * 1024 * 1024,
        # force_final_distributed_eval=True
    )
