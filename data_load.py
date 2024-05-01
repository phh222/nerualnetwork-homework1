import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import json

def load_mnist(path, kind='train'):
    """加载MNIST数据"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

# # 调整这个路径到您的数据存放位置
# path = 'data'

# # 加载数据
# X_train, Y_train = load_mnist(path, kind='train')
# X_test, Y_test = load_mnist(path, kind='t10k')

# # 正常化数据
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# # 划分训练集为训练集和验证集
# from sklearn.model_selection import train_test_split

# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


# train(nn,X_train, Y_train, X_val, Y_val, epochs = 50, learning_rate_decay=0.95)

# nn.params = nn.best_params

# caches = nn.forward(X_test)
# test_accuracy =  calculate_accuracy(caches[-1],Y_test)
# print("Test Accuracy:", test_accuracy)