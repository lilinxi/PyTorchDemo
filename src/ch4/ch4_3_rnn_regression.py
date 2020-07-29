# step1. 训练数据（使用 sin 函数预测 cos 函数）

"""我们使用 x 作为输入的 sin 值, 然后 y 作为想要拟合的输出, cos 值. 因为他们两条曲线是存在某种关系的, 所以我们就能用 sin 来预测 cos. rnn 会理解他们的关系, 并用里面的参数分析出来这个时刻 sin 曲线上的点如何对应上 cos 曲线上的点."""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)  # reproducible

# Hyper Parameters
TIME_STEP = 10  # rnn time step / image height
INPUT_SIZE = 1  # rnn input size / image width
LR = 0.02  # learning rate
DOWNLOAD_MNIST = False  # set to True if haven't download the data


# step2. 建立 RNN 模型

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # torch.Size([1, 10, 1]) -> torch.Size([1, 10, 32])
        self.rnn = nn.RNN(  # 这回一个普通的 RNN 就能胜任
            input_size=1,
            hidden_size=32,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    # def forward(self, x, h_state):  # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state
    #     # x (batch, time_step, input_size)
    #     # h_state (n_layers, batch, hidden_size)
    #     # r_out (batch, time_step, output_size)
    #     r_out, h_state = self.rnn(x, h_state)  # h_state 也要作为 RNN 的一个输入
    #
    #     outs = []  # 保存所有时间点的预测值
    #     for time_step in range(r_out.size(1)):  # 对每一个时间点计算 output
    #         # torch.Size([1, 10, 32]) -> torch.Size([1, 32]) -> torch.Size([1, 1])
    #         outs.append(self.out(r_out[:, time_step, :]))
    #     # [10, torch.Size([1, 1])] -> torch.Size([1, 10, 1])
    #     return torch.stack(outs, dim=1), h_state

    def forward(self, x, h_state):
        # torch.Size([1, 10, 1]) -> torch.Size([1, 10, 32])
        r_out, h_state = self.rnn(x, h_state)
        # torch.Size([1, 10, 32]) -> torch.Size([10, 32])
        r_out = r_out.view(-1, 32)
        # torch.Size([10, 32]) -> torch.Size([10, 1])
        outs = self.out(r_out)
        # torch.Size([10, 1])
        return outs.view(-1, TIME_STEP, 1), h_state


rnn = RNN()
print(rnn)
"""
RNN (
  (rnn): RNN(1, 32, batch_first=True)
  (out): Linear (32 -> 1)
)
"""

# step3. 训练，展示动画

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all rnn parameters
loss_func = nn.MSELoss()

h_state = None  # 要使用初始 hidden state, 可以设成 None

plt.figure(1, figsize=(12, 5))
plt.ion()  # continuously plot

for step in range(100):
    start, end = step * np.pi, (step + 1) * np.pi  # time steps
    # sin 预测 cos
    steps = np.linspace(start, end, 10, dtype=np.float32)
    x_np = np.sin(steps)  # float32 for converting torch FloatTensor
    y_np = np.cos(steps)

    # np.newaxis = None, 增加一个新的坐标轴
    # torch.Size([1, 10, 1])
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)  # rnn 对于每个 step 的 prediction, 还有最后一个 step 的 h_state
    # !!  下一步十分重要 !!
    h_state = h_state.data  # 要把 h_state 重新包装一下才能放入下一个 iteration, 不然会报错

    loss = loss_func(prediction, y)  # cross entropy loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()
