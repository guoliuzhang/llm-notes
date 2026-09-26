import matplotlib.pyplot as plt
import numpy as np

# 用梯度下降拟合带噪声的一元线性数据。
np.random.seed(0)
X = np.random.rand(100, 1) * 5
ow = 7.9
ob = 23
# 噪声均值为 1.5，因此截距预期接近 ob + 1.5。
Y = ow * X + ob + np.random.rand(100, 1) * 3

w = 0.0
b = 0.0
epochs = 100
lr = 0.1

for epoch in range(epochs):
    y = w * X + b

    loss = np.mean((y - Y) ** 2)

    # 对 MSE 求偏导，使用全量样本的平均梯度。
    grad_w = np.mean(2 * (y - Y) * X)
    grad_b = np.mean(2 * (y - Y))

    w -= lr * grad_w
    b -= lr * grad_b

    # 日志中的 w、b 已更新，loss 则来自本轮更新前。
    if epoch % 11 == 0:
        print(f"w is: {w:.4f}, b is: {b:.4f}, loss is: {loss:.4f}")

plt.scatter(X, Y, color="blue", label="data")
plt.plot(X, w * X + b, color="red", label="line")
plt.legend()
plt.show()
