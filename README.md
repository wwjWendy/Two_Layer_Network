# Two_Layer_Network
## 一、项目简介

本项目需要在不调用pytorch、tensorflow等深度学习包的情况下，构建两层神经网络分类器，包括输入层、隐藏层、输入层，使用mnist数据集进行训练并分类。代码中包括了训练、参数查找、测试三个部分，并对最终的训练和测试结果、每层网络的参数进行可视化。

该神经网络的输入层有784（28*28）个神经元，输出层有10个神经元，激活函数使用ReLU，输出层的激活函数使用softmax，并使用numpy包处理矩阵等。



## 二、模型结构

**GitHub地址**：https://github.com/wwjWendy/Two_Layer_Network.git

**模型网盘地址**：链接：https://pan.baidu.com/s/1YbQXqLFKiKZG__F8Y50Odw 提取码：32x4 

（注：下载模型后需将其放在model文件夹中）

 

**dataset**文件夹：包括了mnist.py文件以及已经下载的mnist模型mnist.pkl和mnist数据集的训练集、验证集。其中，mnist.py中包括了下载数据集、将mnist数据集中的图像进行初始化（正则化图像像素值至0-1à展开为一维数组à采用one-hot编码）、以及读入数据集并划分为测试机、验证集的部分。（由于Github内存限制，只上传了mnist.py）

**image**文件夹：包括了程序中输出的所有可视化的结果，包括了使用最优参数进行训练时分别在训练集、测试集上的loss ( loss.svg )、accuracy ( accuracy.svg )，以及可视化每层网络的参数的结果 (W1.svg, W2.svg )。

**model**文件夹：包括了记录寻找最优参数的过程的csv文件 (random_grid_search_results.csv )、使用最优参数进行训练后的模型 ( network.pkl )，以及包含了模型在每个epoch中的训练、验证的loss、accuracy的结果的csv文件 ( loss_acc_epoch_test.csv )。

 

**util.py**：参数查找、训练、可视化loss和accuracy的主程序。在该程序中调用了导入mnist数据集、两层神经网络、训练和超参数优化的类，最终输出参数查找以及训练成果，并将模型进行保存；

**functions.py**：定义了Relu、Affine、SoftmaxWithLoss三个类。Relu类包含了Reluctant函数的前向和后向计算，Affine类包括了前向结果的计算和后向参数梯度的计算，SoftmaxWithLoss.py类中包含了前向计算、计算交叉熵损失、后向求梯度的过程。

**twolayernet.py**：包含了设定权重初始值、根据x和模型得到预测值、求损失函数、求误差率、求梯度的函数。类的输入参数包含了输入的神经元个数、隐藏层的神经元数量的列表、输出层神经元个数、L2范数的强度；

**train.py**：包含了随机梯度下降的类SGD，和训练神经网络的类Trainer，其中训练时使用了mini-batch学习。

**hyperparameter_optimization.py**：使用Random Grid Search进行参数查找，根据测试集accuracy的大小排序最大的情况，返回本轮查找的最有参数的结果。

**test.py**：包括加载训练好的网络，使用该网络对数据进行测试，以及可视化网络中每层的参数的函数。

 

## 三、运行方式

使用本模型时，需要先运行dataset文件夹中的mnist.py文件以下载数据集。下载完毕后，运行util.py，会保存根据查找出的最优参数进行训练的模型，并输出训练过程、最优参数值、loss和accuracy值的可视化的结果。

进行测试时运行test.py，程序会加载上述已经保存的结果，输出loss和accuracy值和可视化的结果、可视化网络中每层的参数的函数。
