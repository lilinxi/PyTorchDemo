# step1. 获取 MNIST 手写数字数据

import torch
import torchvision
import torch.utils.data as Data

torch.manual_seed(1)  # reproducible

# Hyper Parameters
EPOCH = 1  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 64
TIME_STEP = 28  # rnn 时间步数 / 图片高度
INPUT_SIZE = 28  # rnn 每步输入值 / 图片每行像素
LR = 0.01  # learning rate
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 Fasle

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist/',  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,  # 没下载就下载, 下载了就不用再下了
)

# 测试数据
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
         :2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


# step2. 建立 RNN 模型

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # TODO Note：
        # 参数
        # – input_size
        # – hidden_size
        # – num_layers
        # – bias
        # – batch_first
        # – dropout
        # – bidirectional
        # 输入
        # – input (seq_len, batch, input_size)
        # – h_0 (num_layers * num_directions, batch, hidden_size)
        # – c_0 (num_layers * num_directions, batch, hidden_size)
        # 输出
        # – output (seq_len, batch, num_directions * hidden_size)
        # – h_n (num_layers * num_directions, batch, hidden_size)
        # – c_n (num_layers * num_directions, batch, hidden_size)

        # input (batch_size, time_step, input_size->hidden_size)
        # torch.Size([64, 28, 28]) -> torch.Size([64, 28, 64])
        self.rnn = torch.nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
            input_size=28,  # 图片每行的数据像素点
            hidden_size=64,  # rnn hidden unit -> output_size
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = torch.nn.Linear(64, 10)  # 输出层

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        # h_n, h_c -> torch.Size([1, 64, 64])
        # torch.Size([64, 28, 28]) -> torch.Size([64, 28, 64])
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全 0 的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        # torch.Size([64, 28, 64]) -> torch.Size([64, 64])
        # torch.Size([64, 64]) -> torch.Size([64, 10])
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)
"""
RNN (
  (rnn): LSTM(28, 64, batch_first=True)
  (out): Linear (64 -> 10)
)
"""

# step3. 训练

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all parameters
loss_func = torch.nn.CrossEntropyLoss()  # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    print("Epoch:", epoch)
    for step, (x, b_y) in enumerate(train_loader):  # gives batch data
        if step % 100 == 0:
            print("step:", step)

        # torch.Size([64, 1, 28, 28]) -> torch.Size([64, 28, 28])
        b_x = x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
"""
...
Epoch:  0 | train loss: 0.0945 | test accuracy: 0.94
Epoch:  0 | train loss: 0.0984 | test accuracy: 0.94
Epoch:  0 | train loss: 0.0332 | test accuracy: 0.95
Epoch:  0 | train loss: 0.1868 | test accuracy: 0.96
"""

# step4. 预测

test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
"""
[7 2 1 0 4 1 4 9 5 9] prediction number
[7 2 1 0 4 1 4 9 5 9] real number
"""
