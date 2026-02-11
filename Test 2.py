import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

np.random.seed(42)

class Config:
    seed = 42
    num_samples = 5000
    num_classes = 10
    input_dim = 784
    hidden_dim = 256
    repr_dim = 64
    num_epochs = 80
    batch_size = 128
    learning_rate = 0.001
    lambda_invariance = 0.5
    results_dir = './lcrd_results_demo'
    def __init__(self):
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        np.random.seed(self.seed)

config = Config()

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    m = y_true.shape[0]
    return -np.sum(np.log(y_pred[range(m), y_true] + 1e-10)) / m

class SimpleLCRDNetwork:
    def __init__(self, input_dim, hidden_dim, repr_dim, num_classes):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, repr_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, repr_dim))
        self.W3 = np.random.randn(repr_dim, num_classes) * np.sqrt(2.0 / repr_dim)
        self.b3 = np.zeros((1, num_classes))
        self.cache = {}

    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        T = Z2
        T_norm = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-10)
        Z3 = np.dot(T_norm, self.W3) + self.b3
        Y_pred = softmax(Z3)
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'T': T, 'T_norm': T_norm, 'Z3': Z3, 'Y_pred': Y_pred}
        return T_norm, Y_pred

    def backward(self, Y_true, lambda_inv, T_aug):
        m = Y_true.shape[0]
        dZ3 = self.cache['Y_pred'].copy()
        dZ3[range(m), Y_true] -= 1
        dZ3 /= m
        dW3 = np.dot(self.cache['T_norm'].T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        dT_norm = np.dot(dZ3, self.W3.T)
        dT_inv = 2 * (self.cache['T_norm'] - T_aug) * lambda_inv
        dT_total = dT_norm + dT_inv
        dZ2 = dT_total
        dW2 = np.dot(self.cache['A1'].T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.cache['Z1'])
        dW1 = np.dot(self.cache['X'].T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        return {'dW1': dW1,'db1': db1,'dW2': dW2,'db2': db2,'dW3': dW3,'db3': db3}

    def update(self, grads, lr):
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']
        self.W3 -= lr * grads['dW3']
        self.b3 -= lr * grads['db3']

def generate_rotated_mnist_synthetic(num_samples, num_classes, input_dim):
    samples_per_class = num_samples // num_classes
    X_list, Y_list, theta_list = [], [], []
    for c in range(num_classes):
        prototype = np.random.randn(input_dim) * 0.5
        prototype[c*70:(c+1)*70] += 2.0
        for _ in range(samples_per_class):
            sample = prototype + np.random.randn(input_dim) * 0.3
            theta = np.random.rand() * 360
            rotation_effect = np.sin(theta*np.pi/180)*np.random.randn(input_dim)*0.15
            X_list.append(sample + rotation_effect)
            Y_list.append(c)
            theta_list.append(theta)
    X = np.array(X_list); Y = np.array(Y_list); theta = np.array(theta_list)
    X = (X - X.mean(axis=0)) / (X.std(axis=0)+1e-8)
    return X, Y, theta

def train_lcrd_model(model, X, Y, theta, config):
    num_samples = X.shape[0]; num_batches = num_samples // config.batch_size
    history = {'epoch':[],'train_loss':[],'train_acc':[],'task_loss':[],'inv_loss':[],'I_T_Y':[],'I_T_X':[],'I_T_theta':[]}
    for epoch in range(1, config.num_epochs+1):
        idx = np.random.permutation(num_samples)
        X_sh, Y_sh, theta_sh = X[idx], Y[idx], theta[idx]
        loss_sum=task_sum=inv_sum=correct=total=0
        for b in range(num_batches):
            start,end=b*config.batch_size,(b+1)*config.batch_size
            Xb,Yb=X_sh[start:end],Y_sh[start:end]
            T_orig,Y_pred=model.forward(Xb)
            task_loss=cross_entropy(Y_pred,Yb)
            X_aug=Xb+np.random.randn(*Xb.shape)*0.1
            T_aug,_=model.forward(X_aug)
            inv_loss=np.mean((T_orig-T_aug)**2)
            total_loss=task_loss+config.lambda_invariance*inv_loss
            grads=model.backward(Yb,config.lambda_invariance,T_aug)
            lr=config.learning_rate*(1-epoch/config.num_epochs)
            model.update(grads,lr)
            loss_sum+=total_loss; task_sum+=task_loss; inv_sum+=inv_loss
            correct+=np.sum(np.argmax(Y_pred,axis=1)==Yb)
            total+=len(Yb)
        avg_loss=loss_sum/num_batches; avg_task=task_sum/num_batches; avg_inv=inv_sum/num_batches
        acc=100.*correct/total
        history['epoch'].append(epoch); history['train_loss'].append(avg_loss)
        history['train_acc'].append(acc); history['task_loss'].append(avg_task)
        history['inv_loss'].append(avg_inv)
        # placeholder for MI tracking
        history['I_T_Y'].append(0.0); history['I_T_X'].append(0.0); history['I_T_theta'].append(0.0)
    return history

def main():
    X,Y,theta=generate_rotated_mnist_synthetic(config.num_samples,config.num_classes,config.input_dim)
    model=SimpleLCRDNetwork(config.input_dim,config.hidden_dim,config.repr_dim,config.num_classes)
    history=train_lcrd_model(model,X,Y,theta,config)
    print(f"Final accuracy: {history['train_acc'][-1]:.2f}%")
    return history

if __name__=="__main__":
    results=main()
