# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:19:02 2023

@author: 73159
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from twolayernet import TwoLayerNet
from train import Trainer
from hyperparameter_optimization import hyper_param_optimization
import pandas as pd

# 保存训练好的网络
def save_network(network,filename):
    file_name = "model/" + filename
    f=open(file_name,'wb')          
    pickle.dump(network,f)          
    f.close()                  
    print("\n The netowrk has been saved in "+file_name+"!")

# 导入mnist数据集
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
# 测试程序时加快运行速度
#x_train = x_train[:1000]
#t_train = t_train[:1000]

# 超参数优化
print("==== Hyper Parameter Optimization =============================")
np.random.seed(42)
best_hyper_params = hyper_param_optimization([0.05,0.25],[-7,-4],[100,300],50,'model/random_grid_search_results.csv')
#best_hyper_params = hyper_param_optimization([0.05,0.25],[-7,-4],[100,300],1,'test_random_grid_search_results.csv')
#best_hyper_params = {'lr':0.22, 'weight_decay':6.596284392363952e-05, 'hidden_size':179.0}
# 超参数优化的结果
weight_decay_lambda = best_hyper_params['weight_decay']
lr = best_hyper_params['lr']
hidden_size = int(best_hyper_params['hidden_size'])

# 使用最优超参数训练神经网络
print("==== Training with the Optimal Hyper Parameters ===================")
network = TwoLayerNet(input_size=784, hidden_size_list=[hidden_size],
                      output_size=10, weight_decay_lambda=weight_decay_lambda)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=40, mini_batch_size=128,
                  optimizer_param={'lr': lr})
# 训练
trainer.train()
# 保存训练结果
save_network(trainer,'trainer.pkl')
save_network(trainer.network,'network.pkl')
# 记录训练集、测试集在每个epoch的loss和accuracy
test_acc_list_epoch = trainer.test_acc_list_epoch[1:]
train_acc_list_epoch = trainer.train_acc_list_epoch[1:]
train_loss_list_epoch = trainer.train_loss_list_epoch[1:]
test_loss_list_epoch = trainer.test_loss_list_epoch[1:]


# 可视化训练结果
# loss
x1 = np.arange(np.shape(train_loss_list_epoch)[0])
plt.plot(x1, train_loss_list_epoch, label='train')
plt.plot(x1, test_loss_list_epoch, label='test')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, test_loss_list_epoch[0]+0.005)
plt.legend(loc='best')
plt.savefig('image/loss.svg')
plt.show()

# accuracy
x2 = np.arange(np.shape(train_acc_list_epoch)[0])
plt.plot(x2, train_acc_list_epoch, label='train')
plt.plot(x2, test_acc_list_epoch, label='test')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(test_acc_list_epoch[0]-0.005, 1.005)
plt.legend(loc='best')
plt.savefig('image/accuracy.svg')
plt.show()

# 输出训练结果
column_names = ['epoch','train loss','test loss', 'train acc', 'test acc']
loss_acc_df = pd.DataFrame(np.transpose([x2,train_loss_list_epoch,test_loss_list_epoch,train_acc_list_epoch,test_acc_list_epoch]),
                           columns=column_names)
print("\n Train and Test Loss and Accuracy per Epoch: ")
print(loss_acc_df)
loss_acc_df.to_csv('model/loss_acc_epoch_test.csv')
print("\n==== END =========================================")