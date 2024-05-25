# 2021211322 13002109 林哲 《数字媒体智能技术平台》项目2 课程项目作业

import torch
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']#matpoltlib黑体

# 第一步 : 产生随机噪声数据
def dataf(n, k, b):
    # 其中n参数用于rand和randn，用于控制随机函数输出的结果
    x = torch.rand(n) # 产生随机数据，取值范围(0,1]或正态分布(-∞,+∞),数据太大不好处理
    noise = torch.randn(n) * 0.10  # 增加随机少量噪声，权值为0.15,且噪声采样于正态分布内随机数，取值范围(-∞,+∞)
    y = k * x + b + noise # 利用所给线性关系，算出相应数据对
    return x, y # 输出数据对

def datai(n,m,num,k,b):
    # 其中n参数用于rand和randn，用于控制随机函数输出的结果
    x = torch.randint(n,m,(num,1)) * 10 # 产生随机数据，取值整数范围[n,m],乘以系数令权值为0.1
    noise = torch.randint(n,m,(num,1))   # 增加随机少量噪声，噪声采样于取值整数范围[n,m]
    y = k * x + b + noise # 利用所给线性关系，算出相应数据对

    return x, y # 输出数据对
# 第二步 : 通过梯度下降算法来预估k的值和b的值
def torch_grad_des(x, y,k_rec,b_rec,iterations,a=0.06):
    # 学习率a，iterations是迭代次数

    k = torch.rand(1, requires_grad=True) # requires_grad=True 的意思是保存算出的梯度，让optimizer更新参数
    b = torch.rand(1, requires_grad=True)

    # k = torch.randint(1,5,(1,1), requires_grad=True)
    # b = torch.randint(1,5,(1,1), requires_grad=True)


    for iteration in range(iterations):
        # 向前传播，预测y的值
        y_pre = k * x + b

        # 计算并输出损失
        loss = torch.mean((y_pre - y) ** 2) # 这里** 2 就是平方的意思,mean用于计算平均数
        lossMtx = (y_pre - y) ** 2
        if iteration % 100 == 0:  # 每迭代100次输出迭代次数和这100次迭代的平均损失
            # print('Epoch-迭代次数 {}, LossMatrix-损失矩阵{}，Loss-平均损失 {}'.format(epoch, lossMtx,loss.item()))
            print('Epoch-迭代次数 {}, Loss-平均损失 {}'.format(iteration, loss.item()))
            # print('本轮迭代前k {}, 本轮迭代前b{}'.format(k, b))

        # 反向传播来计算k和b的梯度
        loss.backward()
        kr=k
        br=b
        k_rec[iteration] = (float)(kr.detach())
        b_rec[iteration] = (float)(br.detach())

        # 通过计算出来的梯度更新k和b参数值,此时禁用下面张量的梯度计算
        with torch.no_grad():
            k -= a * k.grad
            b -= a * b.grad


        # 当optimizer更新完k和b的值后，手动将k和b的本次梯度归零
        k.grad.zero_()
        b.grad.zero_()
    if iteration==iterations:
        k_rec[iteration] = (float)(kr.detach())
        b_rec[iteration] = (float)(br.detach())
    return k.item(), b.item()


# 产生训练数据

num = 100  # 100个噪声样本
iterations=1000 # 迭代次数
k_true = 2.0  # k真值
b_true = 1.0  # b真值

# x_train, y_train = generate_datai(n,m,num, k_true, b_true)  # 根据k和b的真值以及噪声，生成x，y噪声数据对-100
x_train, y_train = dataf(num,k_true,b_true)  # 根据k和b的真值以及噪声，生成x，y噪声数据对100个

k_rec=[0]*iterations
b_rec=[0]*iterations

# 通过梯度下降算法预估k和b的值
k_estimate, b_estimate = torch_grad_des(x_train,y_train,k_rec,b_rec,iterations)

print('预计的k值:', k_estimate)
print('预估的b值:', b_estimate)

# 输出拟合折线图，横坐标迭代次数，纵坐标k和b的值,黄色为k，蓝色为b
# print(k_rec,b_rec)
plt.plot(range(0, iterations), k_rec, color='yellow', lw=2, marker='o', linestyle='--')
plt.plot(range(0, iterations), b_rec, color='blue', lw=2, marker='o', linestyle='--')
plt.xlabel('迭代次数(绿色为b，红色为=k)'), plt.ylabel('k和b的预估值'), plt.xlim(0, iterations * 1.2), plt.ylim(0,5), plt.show()
