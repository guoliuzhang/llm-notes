import matplotlib.pyplot as plt
import numpy as np

# 一元线性回归示例：生成带噪声的数据，用梯度下降拟合 y = w * X + b。
# 1. 生成训练数据，基础直线为 y = 7.9x + 23。
# 固定随机种子，使每次运行生成相同的数据，便于复现实验。
np.random.seed(0)
# 100 个样本，每个样本只有一个特征；X 的形状为 (100, 1)，取值范围为 [0, 5)。
X = np.random.rand(100, 1) * 5
ow = 7.9  # 生成数据时使用的真实斜率。
ob = 23  # 生成数据时使用的真实截距。
# 加入 [0, 3) 的均匀随机噪声，模拟观测误差。
# 噪声均值为 1.5，因此充分拟合后，截距通常接近 24.5，而不是 23。
Y = ow * X + ob + np.random.rand(100, 1) * 3

# 2. 初始化待学习的参数 w、b，并设置训练轮数和学习率。
w = 0.0
b = 0.0
epochs = 100  # 遍历全部训练样本并更新参数的次数。
lr = 0.1  # 学习率：控制每次沿负梯度方向更新参数的步长。

# 每轮使用全部 100 个样本计算梯度，即批量梯度下降。
for epoch in range(epochs):
    # 根据当前参数计算预测值，y 与真实观测值 Y 的形状相同。
    y = w * X + b

    # 均方误差（MSE）：所有样本预测误差的平方的平均值，越小表示拟合越好。
    loss = np.mean((y - Y) ** 2)

    # 对 MSE 求偏导：dy/dw = X，dy/db = 1，平方项求导得到系数 2。
    grad_w = np.mean(2 * (y - Y) * X)
    grad_b = np.mean(2 * (y - Y))

    # 沿梯度的反方向更新参数，以减小损失。
    w -= lr * grad_w
    b -= lr * grad_b

    # 从第 0 轮起每隔 11 轮输出一次；w、b 已更新，loss 是本轮更新前的值。
    if epoch % 11 == 0:
        print(f"w is: {w:.4f}, b is: {b:.4f}, loss is: {loss:.4f}")

# 3. 将训练样本与最终拟合直线画在同一张图上，直观比较拟合效果。
plt.scatter(X, Y, color="blue", label="data")
plt.plot(X, w * X + b, color="red", label="line")
plt.legend()  # 显示图例，区分样本点和拟合直线。
plt.show()  # 显示图像窗口。
