# step1.  我们创建一些假数据来模拟真实的情况. 比如一个一元二次函数: y = a * x^2 + b, 我们给 y 数据加上一点噪声来更加真实的展示它.

import torch
import matplotlib.pyplot as plt

# unsqueeze 升纬，squeeze 降维
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100,) -> (100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

# 画图
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

# step2.  建立神经网络（两层线性神经元，中间一次 relu 激活）

import torch.nn.functional as F  # 激励函数都在这


class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        # class Linear(Module):     def __init__(self, in_features, out_features, bias=True):
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        x = self.predict(x)  # 输出值
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)  # net 的结构
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""

# step3. 训练网络

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)

# TODO Note: optimizer 和 loss 没有交互？
# loss.backward() 获得所有 parameter 的 gradient。然后 optimizer 存了这些 parameter 的指针，step() 根据这些 parameter 的 gradient 对 parameter 的值进行更新。
# loss.backward()  # 每次计算梯度的时候，其实是有一个动态的图在里面的，求导数就是对图中的参数w进行求导的过程，即所有计算 loss 的节点，构成了一个有向图，可以实现链式求导，即自动求导

# for t in range(100):
#     prediction = net(x)  # 喂给 net 训练数据 x, 输出预测值
#
#     loss = loss_func(prediction, y)  # 计算两者的误差
#
#     optimizer.zero_grad()  # 清空上一步的残余更新参数值
#     loss.backward()  # 误差反向传播, 计算参数更新值
#     optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

# step4. 可视化训练过程

import matplotlib.pyplot as plt

plt.ion()  # 画图，打开交互模式
plt.show()

for t in range(200):
    prediction = net(x)  # 喂给 net 训练数据 x, 输出预测值，本质是调用 forward()
    loss = loss_func(prediction, y)  # 计算两者的误差，loss：<class 'torch.Tensor'>
    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 每次计算梯度的时候，其实是有一个动态的图在里面的，求导数就是对图中的参数w进行求导的过程，即所有计算 loss 的节点，构成了一个有向图，可以实现链式求导，即自动求导
    optimizer.step()

    # 接着上面来
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()  # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变。（cla 清除 axes，clf 清除 figure，close 清除 windows）
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
