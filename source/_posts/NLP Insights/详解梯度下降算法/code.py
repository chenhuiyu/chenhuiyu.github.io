import numpy as np

# 设定随机数种子
np.random.seed(0)

# 输入数据
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

# 输出数据
y = np.array([[0,1,1,0]]).T

# sigmoid函数
def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

# 初始化权重
w0 = 2*np.random.random((3,4)) - 1
w1 = 2*np.random.random((4,1)) - 1

# 迭代训练
for j in range(60000):

    # 前向传播
    l0 = X
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    # 计算损失
    l2_error = y - l2

    # 打印损失值
    if j % 10000 == 0:
        print(f'Error: {np.mean(np.abs(l2_error))}')

    # 反向传播
    l2_delta = l2_error * sigmoid(l2, deriv=True)
    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * sigmoid(l1, deriv=True)

    # 更新权重
    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)
