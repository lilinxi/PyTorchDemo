"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
numpy
matplotlib
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible
# np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
# 灵感，随机数的个数
N_IDEAS = 5  # think of this as number of ideas for generating an art work (Generator)
# 作品，模拟数的个数
ART_COMPONENTS = 15  # it could be total point G can draw in the canvas
# 一批数据 (64, 15)：真实数据或模拟数据
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


# 需要模拟的正确的模型，一个上届和一个下届
def artist_works():  # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return paintings


# show our beautiful painting range
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()

G = nn.Sequential(  # Generator
    nn.Linear(N_IDEAS, 128),  # random ideas (could from normal distribution)
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),  # making a painting from these random ideas
)

D = nn.Sequential(  # Discriminator
    nn.Linear(ART_COMPONENTS, 128),  # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),  # tell the probability that the art work is made by artist
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()  # something about continuous plotting

for step in range(10000):
    # step1. 拿到一批真实的数据模型
    artist_paintings = artist_works()  # real painting from artist
    # step2. 随机生成一批模拟的数据模型
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)  # random ideas\n
    G_paintings = G(G_ideas)  # fake painting from G (random ideas)
    # step3. 提示模拟技术，以'鉴别为假'为损失函数
    prob_artist1 = D(G_paintings)  # D try to reduce this prob
    G_loss = torch.mean(torch.log(1. - prob_artist1))
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    # step4. 真假比较，提升鉴别技术，以'分清真假'为损失函数
    prob_artist0 = D(artist_paintings)  # D try to increase this prob
    # TODO Note:
    # GAN的G的更新，主要是GAN loss。就是G生成的fake图让D来判别，得到的损失，计算梯度进行反传。这个梯度只能影响G，不能影响D!
    # detach 可以截断反向传播的梯度流
    prob_artist1 = D(G_paintings.detach())  # D try to reduce this prob
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    opt_D.zero_grad()
    # TODO Note:
    # 一个梯度在两个地方使用，需要保存数据，默认不保存，可以加速（在这里好像并不起作用）
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()
