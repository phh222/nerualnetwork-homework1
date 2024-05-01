from sklearn.model_selection import ParameterGrid
from train import train
from data_load import load_mnist
from model import NeuralNetwork

def grid_search_hyperparameters(param_grid, X_train, Y_train, X_val, Y_val):
    best_accuracy = float('inf')
    best_params = None
    best_model = None

    # 创建参数组合的网格
    grid = ParameterGrid(param_grid)
    for params in grid:
        # 初始化模型
        model = NeuralNetwork(hidden_size1=params['hidden_size1'],
                              hidden_size2=params['hidden_size2'],
                              activations=params['activations'],
                              lambda_reg=params['lambda_reg'],
                              learning_rate=params['learning_rate'])
        print(f"Testing model with params: {params}")
        # 训练模型
        train(model, X_train, Y_train, X_val, Y_val,params = params, epochs=50)
        
        # 检查这个模型的验证损失是否是最佳
        if model.best_accuracy > best_accuracy:
            best_accuracy = model.best_accuracy
            best_params = params
            best_model = model

    return best_model, best_params

# 超参数网格定义
param_grid = {
    'hidden_size1': [256, 128],
    'hidden_size2': [64, 32],
    'activations': [['relu', 'relu'],['sigmoid','sigmoid']],
    'lambda_reg': [0.001, 0.01],
    'learning_rate': [0.01, 0.05,0.1]
}

# 调整这个路径到您的数据存放位置
path = 'data'

# 加载数据
X_train, Y_train = load_mnist(path, kind='train')
X_test, Y_test = load_mnist(path, kind='t10k')

# 正常化数据
X_train = X_train / 255.0
X_test = X_test / 255.0

# 划分训练集为训练集和验证集
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

print("Start search!")

# 执行超参数搜索
best_model, best_params = grid_search_hyperparameters(param_grid, X_train, Y_train, X_val, Y_val)
print(f"Best params: {best_params}, with best validation loss: {best_model.best_loss}")

# 保存最佳模型的参数
output_dir = "best_model_params/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
best_model.save_best_weights(os.path.join(output_dir, "best_model.json"))
