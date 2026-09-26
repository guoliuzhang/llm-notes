# 用线性层学习 y = 3x + 5。
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 每行一个样本，保留特征维度：[样本数, 1]。
data = torch.tensor(
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]],
    dtype=torch.float32,
)
labels = data * 3 + 5

dataset = SimpleDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = SimpleModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 100000
lr = 0.00001

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

for epoch in range(epochs):
    model.train()
    for inputs, labels in dataloader:
        # 梯度默认累积，每个批次前清零。
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    # loss 是最后一个批次更新前的损失。
    if epoch % ((epochs - 1) // 9) == 0:
        print(f"epoch [{epoch}/{epochs}], loss is {loss.item():.8f}")

test_data = torch.tensor([[21.0]], dtype=torch.float32).to(device)

model.eval()
# eval() 不关闭梯度记录，推理时另用 no_grad()。
with torch.no_grad():
    result = model(test_data)
    print(
        f"result is {result.item():.4f}, real result is {(test_data * 3 + 5).item():.4f}"
    )
