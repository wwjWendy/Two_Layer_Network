# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 23:01:03 2023

@author: 73159
"""


import numpy as np
import pandas as pd
from dataset.mnist import load_mnist
from twolayernet import TwoLayerNet
from train import Trainer

# 打乱训练、预测数据集
def shuffle_dataset(x, t):
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]
    return x, t


# 给定超参数后进行训练
def train_with_hyper_param(x_train, t_train, x_val, t_val, 
                           lr, weight_decay, hidden_size, epocs=50):
    network = TwoLayerNet(input_size=784, hidden_size_list=[hidden_size],
                                output_size=50, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                    epochs=epocs, mini_batch_size=128,
                    optimizer_param={'lr': lr})
    trainer.train()
    return trainer.test_acc


# 超参数优化：随机网格搜索
# lr_interval：学习率的搜索范围；weight_decay_interval：L2范数强度的搜索范围（指数）
# hidden_size_interval：隐藏层宽度的搜索范围；optimization_trial：随机搜索的次数
def hyper_param_optimization(lr_interval,weight_decay_interval,hidden_size_interval,
                       optimization_trial = 100,result_file = 'model/rgs_results.csv'):
    # 载入数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    # 测试程序时，为加快速度，截取一部分训练数据
    # x_train = x_train[:1000]
    # t_train = t_train[:1000]

    # 分割验证集并打乱
    validation_rate = 0.20
    validation_num = int(x_train.shape[0] * validation_rate)
    x_train, t_train = shuffle_dataset(x_train, t_train)
    x_val = x_train[:validation_num]
    t_val = t_train[:validation_num]
    x_train = x_train[validation_num:]
    t_train = t_train[validation_num:]

    # 超参数的随机搜索
    lr_list = []
    weight_decay_list = []
    hidden_size_list = []
    val_acc_list = []
    for i in range(optimization_trial):
        print("---- Random Trial " + str(i+1) + " -----------------")
        # 指定搜索的超参数的范围
        weight_decay = 10**np.random.uniform(weight_decay_interval[0],weight_decay_interval[1])
        lr = np.random.uniform(lr_interval[0], lr_interval[1])
        hidden_size = np.random.randint(hidden_size_interval[0],hidden_size_interval[1])
        print(" lr = " + str(round(lr,6)) + 
              ", weight decay = " + str(round(weight_decay,7)) + 
              ", hidden layer size = " + str(hidden_size))
        
        # 训练
        val_acc = train_with_hyper_param(x_train, t_train, x_val, t_val, 
                                         lr, weight_decay, hidden_size)
        print(" random trial "  + str(i+1) + 
              ": val acc = " + str(round(val_acc,6)))
        lr_list.append(lr)
        weight_decay_list.append(weight_decay)
        hidden_size_list.append(int(hidden_size))
        val_acc_list.append(val_acc)
        
    # Random Grid Search 的结果
    print("\n==== Hyper-Parameters Ranking List =======================")
    grid_search_results = pd.DataFrame(np.transpose([lr_list,weight_decay_list,hidden_size_list,val_acc_list]),
                                       columns=['lr','weight decay','hidden size','val acc'])
    grid_search_results = grid_search_results.sort_values(by='val acc',ascending=False, ignore_index=True)
    grid_search_results.to_csv(result_file)
    print(grid_search_results)
    best_hyper_params = list(grid_search_results.loc[0,'lr':'hidden size'])
    best_hyper_params = dict(zip(['lr','weight_decay','hidden_size'],best_hyper_params))
    return best_hyper_params

if __name__ == "__main__":
    best_hyper_params = hyper_param_optimization([0.1,0.5],[-7,-3],[30,50],5,'model/test_rgs_results.csv')
    print(best_hyper_params)