from model import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import json
from data_load import load_mnist

def generate_directory_name(params):
    """ 根据超参数生成目录名称。"""
    # 创建一个基于超参数的名称，例如 "HS1-100_HS2-50_LR-0.01"
    dir_name = f"HS1-{params['hidden_size1']}_HS2-{params['hidden_size2']}_LR-{params['learning_rate']}_REG-{params['lambda_reg']}"
    return dir_name

def train(model, X_train, Y_train, X_val, Y_val, epochs,params, batch_size=64, learning_rate_decay=0.98):
    n_train_samples = X_train.shape[0]
    n_val_samples = X_val.shape[0]
    training_log = {"epochs": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(n_train_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        
        # Training phase
        total_train_loss = 0.0
        for start in range(0, n_train_samples, batch_size):
            
            end = min(start + batch_size, n_train_samples)
            X_batch = X_train_shuffled[start:end]
            Y_batch = Y_train_shuffled[start:end]
            
            caches = model.forward(X_batch)
            train_loss = model.compute_loss(caches[-1], Y_batch)
            
            total_train_loss += train_loss
            
            grads = model.backward(X_batch,caches, Y_batch)
            # model.update_params(grads)
        
        avg_train_loss = total_train_loss / (n_train_samples // batch_size)
        
        # Validation phase

        val_caches = model.forward(X_val)
        avg_val_loss = model.compute_loss(val_caches[-1],Y_val)
        val_accuracy = model.accuracy(val_caches[-1],Y_val)
        
        # Update best model
        if val_accuracy > model.best_accuracy:
            model.best_accuracy = val_accuracy
            model.best_params = model.params.copy()  # Save the best model parameters
        
        # Learning rate decay
        model.learning_rate *= learning_rate_decay

        # Log training progress
        training_log["epochs"].append(epoch + 1)
        training_log["train_loss"].append(float(avg_train_loss))
        training_log["val_loss"].append(float(avg_val_loss))
        training_log["val_accuracy"].append(float(val_accuracy))
        

        print(f'Epoch {epoch + 1}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}')
    # 创建目录

    specific_dir = generate_directory_name(params)
    output_dir = os.path.join('models', specific_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save best model parameters
    model.params = model.best_params
    model.save_best_weights(os.path.join(output_dir,"model.json"))

    # Save training log
    with open(os.path.join(output_dir,"log.json"), 'w') as f:
        json.dump(training_log, f)
    

    # Plot training loss curve and save
    plt.figure(figsize=(8, 6))
    plt.plot(training_log["epochs"], training_log["train_loss"], label="Training Loss", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir,"loss_curve.png"))
    plt.show()

    # Plot validation accuracy curve and save
    plt.figure(figsize=(8, 6))
    plt.plot(training_log["epochs"], training_log["val_accuracy"], label="Validation Accuracy", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.show()


# nn = NeuralNetwork(**params)

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


# train(nn,X_train, Y_train, X_val, Y_val, epochs = 50, learning_rate_decay=0.98)

# nn.params = nn.best_params

