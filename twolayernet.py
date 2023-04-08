# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 19:53:03 2023

@author: 73159
"""


import numpy as np
from collections import OrderedDict
from functions import *

# 两层神经网络的类
class TwoLayerNet:
    """
    全连接的两层神经网络
    ----------------------------------
    输入：
    input_size : 输入大小（MNIST下为784）
    hidden_size_list : 隐藏层的神经元数量的列表，可以不是两层（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST下为10）
    activation : 激活函数'relu' or 'sigmoid'
    weight_init_std : 
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : L2范数的强度
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 初始化权重
        self.__init_weight(weight_init_std)
        # 生成层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])
        self.last_layer = SoftmaxWithLoss()


    # 设定权重的初始值
    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])


    # 求预测值
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x


    # 求损失函数，其中x为预测值，t为真实值
    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
        return self.last_layer.forward(y, t) + weight_decay


    # 求误差率，其中x为预测值，t为真实值
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    # 求梯度（误差反向传播法），其中x为预测值，t为真实值
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 返回具有各层参数梯度的字典变量（如grad['W1']表示W1内部各参数的梯度）
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
        return grads