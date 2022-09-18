# Flower 進階練習
使用 mnist data 來預測數字吧
> 初始練習-[第一次接觸 Flower 就上手](https://hackmd.io/@GvGUX7NOQlezhmIjgNA4tw/Hy9Lfnrej)
## 硬體設備及環境
- 作業系統 Windows11 專業版 21H2
- 中央處理器 AMD Ryzen 5 5600X CPU @ 3.70GHz
- 記憶體 16GB
- 開發工具 PyCharm Community Edition 2022.2.1
- 程式語言 Python 3.9.9 (需要 Python 3.7 或以上)
- 套件
    - flower 0.19.0
    - numpy 1.20.0
    - tensorflow 2.7.0
    - matplotlib 3.5.1
    - keras 2.7.0
    - seaborn 0.11.2

## 初次使用安裝指令
```
python -m pip install --upgrade pip 
pip install flwr
pip install tensorflow
pip install numpy
pip install matplotlib
pip install keras
pip install seaborn
```

## Client 端
> [參考影片](https://youtu.be/3GIb707Yj8k)
> 模型來源不可考

首先新增檔名為 `client.py`，並匯入會使用的套件
```python=
import flwr as fl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils
import seaborn as sns
``` 

### 創建 Model
```python=37
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, input_shape=(1, 28, 28), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### 切分資料
我們使用 mnist 資料集，並切分成 training 及 testing dataset。預設有 60000 筆 training 及 10000 筆 testing data。每張圖片由 28 * 28 個像素組成，將28 * 28的圖片矩陣轉為 784 個數字，並做正規化。
> 像素最大值為 255
```python=46
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 1, 28, 28)/255
x_test = x_test.reshape(10000, 1, 28, 28)/255
```
這裡須注意的是，由於要體現「聯邦學習」，兩個 client 的訓練資料並不相同。
> client1
```python=50
dist = [4000, 4000, 4000, 3000, 10, 10, 10, 10, 4000, 10]
```
> client2
```python=50
dist = [0, 10, 10, 10, 4000, 3000, 4000, 5000, 10, 4500]
```
之後我們依照剛剛設定好的資料量使用`getData`拿資料，使用`getDist`將資料量使用圖表顯示出來，由於 client2 沒有拿到數字 0 的資料，因此 plot 會少一行。 
```python=17
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
```
```python=51
x_train, y_train = getData(dist, x_train, y_train)
getDist(y_train)
```

將 label 轉為僅有 01 的矩陣（One Hot Encoding）
```python=53
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
```
到這邊 model 跟資料集設定好了，接下來跟之前一樣設定客戶端並啟動

```python=57
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters) # update the parameters from the server
        r = model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

fl.client.start_numpy_client(
    "localhost:7001",
    client=CifarClient()
)
```

## Server 端
這邊與上個練習一樣啟動 server，不過我們設定聚合所使用的策略為 FedAvg，學習率設為 0.1 。
若沒有設定初始 model，Server 會隨機選一個客戶端做為第一輪的全域模型，所以也可以自己設定初始 model。這邊所使用初始 model 與 client 端的架構一致，但未做任何訓練。

```python=14
#Configuring client fit and client evaluate
def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    def fit_config(rnd: int) -> Dict[str, str]:
        config = {
            "learning_rate": str(0.001),
            "batch_size": str(32),
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
        fraction_fit=0.1,  #學習率
        min_available_clients=2,  #最少要幾個 client 才能進行訓練
        # initial_parameters=weights_to_parameters(model.get_weights()),  #設定初始model
        # on_fit_config_fn=get_on_fit_config_fn(),  #可以自定義在 fitting 時所使用的 參數，
    )

    # Start Flower server
    fl.server.start_server(
        server_address="localhost:7001",
        config={"num_rounds": 3},  #整個聚合過程執行三次
        strategy=strategy,
        # force_final_distributed_eval=True  #將結果完整顯示出來，用於 debug
    )
```
## 實驗結果
由於兩個 Client 端使用的訓練資料不同，訓練出的 Model 也不同，因此 Accuracy 也不同，不過 Testing data 相同，Server 在合併完 weights 後，兩者在驗證時的 Accuracy 會相同，且 Validate 的 Accuracy也有增加。
第 1、3、5 次訓練為 Client 端的訓練結果，第 2、4、6 次為聚合後的 Global Model 傳回 Client 端做 evaluate 的結果，第 7 次為訓練結束後使用 Global Model 到各個 Client 來做 evaluate 的結果。由於兩個 Client 端的驗證測資相同，所以跑出相同結果。

### Client1
![](https://i.imgur.com/JILffZU.png)
### Client2
![](https://i.imgur.com/IMGE3uH.png)

[程式碼](https://github.com/sheway/Flower_mnist)
