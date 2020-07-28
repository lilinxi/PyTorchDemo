import torch
import matplotlib.pyplot as plt

# TODO Note: 运行这段代码会发现，每次得到的随机数是固定的。但是如果不加上torch.manual_seed这个函数调用的话，打印出来的随机数每次都不一样。
# torch.manual_seed(1) 训练不出来？？？？？
torch.manual_seed(2)  # reproducible

# 假数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)


# step1. 保存网络结构和网络参数

def save():
    # 建网络
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    # 训练
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    prediction = net1(x)
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    torch.save(net1, 'net.pkl')  # 保存整个网络
    torch.save(net1.state_dict(), 'net_params.pkl')  # 只保存网络中的参数 (速度快, 占内存少)


# step2. 读取整个网络

def restore_net():
    # restore entire net1 to net2
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    plt.plot(x.data.numpy(), prediction.data.numpy() + 0.1, 'g-', lw=5)


# step3. 读取网络参数

def restore_params():
    # 新建 net3
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # 将保存的参数复制到 net3
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    plt.plot(x.data.numpy(), prediction.data.numpy() + 0.2, 'b-', lw=5)


# step4. 展示三个网络的预测结果（有一点偏移）

# 保存 net1 (1. 整个网络, 2. 只有参数)
save()

# 提取整个网络
restore_net()

# 提取网络参数, 复制到新网络
restore_params()

plt.show()
