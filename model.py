import numpy as np
import json
class NeuralNetwork:
    def __init__(self,  hidden_size1, hidden_size2,activations,lambda_reg,learning_rate,input_size=784,output_size=10):
        self.activations = activations
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.params = {}
        # Initialize parameters
        self.params['W1'] = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
        self.params['b1'] = np.zeros((1, hidden_size1))
        self.params['W2'] = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
        self.params['b2'] = np.zeros((1, hidden_size2))
        self.params['W3'] = np.random.randn(hidden_size2, output_size) * np.sqrt(2. / hidden_size2)
        self.params['b3'] = np.zeros((1, output_size))
        self.best_accuracy = 0  # 初始化为正无穷
        self.best_params = {}  # 初始化为空字典


    def activation_function(self, x, act_type):
        if act_type == 'relu':
            return np.maximum(0, x)
        elif act_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif act_type == 'softmax':
            e_x = np.exp(x)
            return e_x / np.sum(e_x, axis=1, keepdims=True)
        return x

    def forward(self, X):
        
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.activation_function(Z1, self.activations[0])
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.activation_function(Z2, self.activations[1])
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        A3 = self.activation_function(Z3, 'softmax')  # Output layer always uses softmax
        
        return A1, A2, A3

    def compute_loss(self, Y_hat, Y):
        
        m = Y.shape[0]
        
        # print(Y_hat.shape,Y.shape)
        log_likelihood = [-np.log(Y_hat[i,y] + 1e-8) for i,y in enumerate(Y)]
        loss = np.sum(log_likelihood) / m
        # Add L2 regularization
        l2_cost = (self.lambda_reg / (2 * m)) * (np.sum(np.square(self.params['W1'])) +
                                            np.sum(np.square(self.params['W2'])) +
                                            np.sum(np.square(self.params['W3'])))
        return loss + l2_cost

    def backward(self,X, caches, Y):
        A1, A2, A3 = caches
        m = Y.shape[0]
        Y_one_hot = np.eye(self.params['W3'].shape[1])[Y]

        dZ3 = A3 - Y_one_hot
        dW3 = (A2.T.dot(dZ3) + self.lambda_reg * self.params['W3']) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dZ2 = dZ3.dot(self.params['W3'].T) * (A2 > 0)  # Assuming relu
        dW2 = (A1.T.dot(dZ2) + self.lambda_reg * self.params['W2']) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = dZ2.dot(self.params['W2'].T) * (A1 > 0)  # Assuming relu
        dW1 = (X.T.dot(dZ1) + self.lambda_reg * self.params['W1']) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update parameters with SGD
        self.params['W1'] -= self.learning_rate * dW1
        self.params['b1'] -= self.learning_rate * db1
        self.params['W2'] -= self.learning_rate * dW2
        self.params['b2'] -= self.learning_rate * db2
        self.params['W3'] -= self.learning_rate * dW3
        self.params['b3'] -= self.learning_rate * db3


    def predict(self, X):
        _, _, output = self.forward(X)
        return np.argmax(output, axis=1)

    def accuracy(self, Y_pred, Y_true):
        predictions = np.argmax(Y_pred, axis=1)
        correct_count = np.sum(predictions == Y_true)
        total_samples = len(Y_true)
        accuracy = correct_count / total_samples

        return accuracy

    def save_weights(self, file_path):
        serializable_params = {k: v.tolist() for k, v in self.params.items()}
        with open(file_path, 'w') as f:
            json.dump(serializable_params, f)

    def save_best_weights(self, file_path):
        serializable_best_params = {k: v.tolist() for k, v in self.best_params.items()}
        with open(file_path, 'w') as f:
            json.dump(serializable_best_params, f)

    def load_weights(self, file_path):
        with open(file_path, 'r') as f:
            loaded_params = json.load(f)
        self.params = {k: np.array(v) for k, v in loaded_params.items()}