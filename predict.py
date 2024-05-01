import numpy as np

from model import NeuralNetwork
from data_load import load_mnist
import gzip

import matplotlib.pyplot as plt
import numpy as np

def plot_weights(model):
    w1 = model.params['W1']
    num_filters = w1.shape[1]
    num_grids = int(np.ceil(np.sqrt(num_filters)))
    
    fig, axes = plt.subplots(num_grids, num_grids, figsize=(num_grids*1.5, num_grids*1.5))
    
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w1[:, i].reshape(28, 28)  # 假设输入图像是28x28像素
            ax.imshow(img, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig("weight1.png")
    plt.show()
    



def plot_activations(model, X_sample):
    _, activations, _ = model.forward(X_sample)
    
    num_filters = activations.shape[1]
    num_grids = int(np.ceil(np.sqrt(num_filters)))
    
    fig, axes = plt.subplots(num_grids, num_grids, figsize=(num_grids*1.5, num_grids*1.5))
    
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            ax.imshow(activations[0, :, i].reshape(28, 28), cmap='viridis')  # 假设第一层激活可以reshape成28x28
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    



def plot_weight_distribution(model):
    weights = [model.params[key] for key in model.params.keys() if 'W' in key]
    plt.figure(figsize=(12, 6))
    for i, weight in enumerate(weights):
        plt.subplot(1, len(weights), i + 1)
        plt.hist(weight.ravel(), bins=50)
        plt.title(f'Layer {i+1} weights')
    plt.show()
    plt.savefig("weight_distribution.png")


nn = NeuralNetwork(hidden_size1=256,hidden_size2=32,activations=["relu","relu"],lambda_reg=0.001,learning_rate=0.1)
nn.load_weights("model.json")

# 调整这个路径到您的数据存放位置
path = 'data'

# 加载数据
X_test, Y_test = load_mnist(path, kind='t10k')

# 正常化数据
X_test = X_test / 255.0

caches = nn.forward(X_test)
test_accuracy =  nn.accuracy(caches[-1],Y_test)
print("Test Accuracy:", test_accuracy)
# plot_weights(nn)  # 假设 nn 是你的模型实例
plot_weight_distribution(nn)
