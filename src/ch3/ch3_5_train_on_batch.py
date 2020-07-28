import torch
import torch.utils.data as Data

if __name__ == '__main__':
    # TODO Note: 必须在主函数里，这样，之后才能 fork 多线程
    # 为使用了 multiprocessing  的程序，提供冻结以产生 Windows 可执行文件的支持。
    # 需要在 main 模块的 if __name__ == '__main__' 该行之后马上调用该函数。
    # 由于Python的内存操作并不是线程安全的，对于多线程的操作加了一把锁。这把锁被称为GIL（Global Interpreter Lock）。
    # 而 Python 使用多进程来替代多线程
    torch.multiprocessing.freeze_support()

    torch.manual_seed(1)  # reproducible

    BATCH_SIZE = 3  # 批训练的数据个数

    x = torch.linspace(1, 10, 10)  # x data (torch tensor)
    y = torch.linspace(10, 1, 10)  # y data (torch tensor)

    # 先转换成 torch 能识别的 Dataset (data_tensor=x, target_tensor=y)
    torch_dataset = Data.TensorDataset(x, y)

    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )

    for epoch in range(3):  # 训练所有!整套!数据 3 次
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
            # 假设这里就是你训练的地方...

            # 打出来一些数据
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())

    """
    Epoch:  0 | Step:  0 | batch x:  [ 6.  7.  2.  3.  1.] | batch y:  [  5.   4.   9.   8.  10.]
    Epoch:  0 | Step:  1 | batch x:  [  9.  10.   4.   8.   5.] | batch y:  [ 2.  1.  7.  3.  6.]
    Epoch:  1 | Step:  0 | batch x:  [  3.   4.   2.   9.  10.] | batch y:  [ 8.  7.  9.  2.  1.]
    Epoch:  1 | Step:  1 | batch x:  [ 1.  7.  8.  5.  6.] | batch y:  [ 10.   4.   3.   6.   5.]
    Epoch:  2 | Step:  0 | batch x:  [ 3.  9.  2.  6.  7.] | batch y:  [ 8.  2.  9.  5.  4.]
    Epoch:  2 | Step:  1 | batch x:  [ 10.   4.   8.   1.   5.] | batch y:  [  1.   7.   3.  10.   6.]
    """
