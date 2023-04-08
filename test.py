# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 17:33:08 2023

@author: 73159
"""


from dataset.mnist import load_mnist
import pickle
from matplotlib import pyplot as plt

# 加载已经训练好的模型
def load_network(filename):
    try:
        file_name = "model/" + filename
        f=open(file_name,'rb')
        network=pickle.load(f)
        f.close()
        return network
    except EOFError:
        return ""


# 载入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
# 加载保存好的网络
network = load_network('network.pkl')
# 计算训练集和测试集的accuracy并输出结果
train_acc = network.accuracy(x_train, t_train)
test_acc = network.accuracy(x_test, t_test)
print("==== Test ===========================" + 
      "\n train_acc: " + str(train_acc) + "\n test_acc: " + str(test_acc))


# 可视化网络的每层的参数
params = network.params
W1, W2 = params['W1'], params['W2']
# W1
plt.figure(figsize=(8, 8))
plt.imshow(W1, cmap='RdBu_r', interpolation='nearest')
plt.ylabel("Input Layer")
plt.xlabel("Hidden Layer")
plt.colorbar(label='Parameter Value')
plt.savefig('image/W1.svg')
plt.show()
# W2
plt.figure(figsize=(8,8))
plt.imshow(W2, cmap='RdBu_r', interpolation='nearest')
plt.xlabel("Output Layer")
plt.ylabel("Hidden Layer")
plt.colorbar(label='Parameter Value')
plt.savefig('image/W2.svg')
plt.show()