import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def get_data_loader(train):
    # train=True 加载训练集，False 加载测试集；ToTensor 将像素缩放到 [0, 1]。
    dataset = MNIST(
        "", train, transform=torchvision.transforms.ToTensor(), download=True
    )
    return DataLoader(dataset, batch_size=15, shuffle=True)


test_data = get_data_loader(False)


def evaluate(test_data, net):
    net.eval()  # 切换到评估模式，与关闭梯度记录是两回事。
    count = 0
    total = 0
    with torch.no_grad():  # 评估时无需记录计算图。
        for x, y in test_data:
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                # 最大输出对应的类别就是预测数字，累计预测正确的样本数。
                if torch.argmax(output) == y[i]:
                    count += 1
                total += 1
    return count / total  # 准确率 = 正确样本数 / 总样本数。


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 将 784 维图像向量经三个隐藏层映射为 10 个数字类别的输出。
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        # 沿类别维度计算对数概率，供下面的 nll_loss 使用。
        return nn.functional.log_softmax(self.fc4(x), dim=1)


net = Net()
criterion = nn.functional.nll_loss
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_data = get_data_loader(True)
# 每轮遍历完整训练集，结束后在测试集上计算准确率。
for epoch in range(3):
    net.train()
    for x, y in train_data:
        optimizer.zero_grad()  # 清除上一批次的梯度，避免累积。
        # 将 [批大小, 1, 28, 28] 展平成 [批大小, 784]。
        output = net.forward(x.view(-1, 28 * 28))
        loss = criterion(output, y)
        loss.backward()  # 反向传播，计算参数的梯度。
        optimizer.step()  # 根据梯度更新参数。
    print("epoch: ", epoch + 1, "accuracy: ", evaluate(test_data, net))

# 从前 6 个测试批次中各取第一张图，展示图像和预测数字。
for i, (x, _) in enumerate(test_data):
    if i > 5:
        break
    predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
    plt.figure(i)
    plt.imshow(x[0].view(28, 28))
    plt.title(f"prediction: {predict.item()}")

plt.show()
