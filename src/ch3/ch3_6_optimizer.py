import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # TODO Note: 必须在主函数里，这样，之后才能 fork 多线程
    # 为使用了 multiprocessing  的程序，提供冻结以产生 Windows 可执行文件的支持。
    # 需要在 main 模块的 if __name__ == '__main__' 该行之后马上调用该函数。
    # 由于Python的内存操作并不是线程安全的，对于多线程的操作加了一把锁。这把锁被称为GIL（Global Interpreter Lock）。
    # 而 Python 使用多进程来替代多线程
    torch.multiprocessing.freeze_support()

    # step1. 制造数据

    torch.manual_seed(1)  # reproducible

    LR = 0.01
    BATCH_SIZE = 32
    EPOCH = 12

    # fake dataset
    x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
    y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

    # plot dataset
    plt.scatter(x.numpy(), y.numpy())
    plt.show()

    # 使用上节内容提到的 data loader
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, )


    # step2. 每个优化器优化一个神经网络

    # 默认的 network 形式
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(1, 20)  # hidden layer
            self.predict = torch.nn.Linear(20, 1)  # output layer

        def forward(self, x):
            x = F.relu(self.hidden(x))  # activation function for hidden layer
            x = self.predict(x)  # linear output
            return x


    # 为每个优化器创建一个 net
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    # step3. 优化器 Optimizer

    # different optimizers
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    losses_his = [[], [], [], []]  # 记录 training 时不同神经网络的 loss

    # step4. 训练

    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        for step, (b_x, b_y) in enumerate(loader):

            # 对每个优化器, 优化属于他的神经网络
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(b_x)  # get output for every net
                loss = loss_func(output, b_y)  # compute loss for every net
                opt.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                opt.step()  # apply gradients
                l_his.append(loss.data.numpy())  # loss recoder

    # step5. 绘制 loss 图
    xAxis = len(losses_his[0])
    plt.plot(torch.linspace(0, 1, xAxis), losses_his[0], 'r-', lw=1, label="SGD")
    plt.plot(torch.linspace(0, 1, xAxis), losses_his[1], 'g-', lw=1, label="Momentum")
    plt.plot(torch.linspace(0, 1, xAxis), losses_his[2], 'b-', lw=1, label="RMSprop")
    plt.plot(torch.linspace(0, 1, xAxis), losses_his[3], 'y-', lw=1, label="Adam")
    plt.legend(loc='best')
    plt.show()
