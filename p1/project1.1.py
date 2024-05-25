# 2021211322 13002109 林哲《数字媒体智能技术平台》项目1 课程项目作业
import random
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#matpoltlib黑体

# 数据生成
def data(n, k, b):
    x = [random.uniform(0, 5) for _ in range(n)] #生成随机数据，范围[0,5]
    y = [k * xi + b + random.uniform(-0.1, 0.1) for xi in x]  # 增加细微噪声，前后加减0,1
    return x, y

# 梯度下降计算loss并拟合
def gradient_des(x, y,k_rec,b_rec,iterations,a=0.06):

    # 初始化参数
    k = random.random()
    b = random.random()

    #记录每一次的k和b变化


    for iteration in range(iterations):

    # 计算y预算值
        y_pred = [k * xi + b for xi in x]

    # 计算loss
        loss = sum((y_pred[i] - yi) ** 2 for i, yi in enumerate(y)) / len(y)
        if iteration % 100 == 0:
            print('迭代次数 {}, 损失 {}'.format(iteration, loss))

        # 计算梯度
        grad_k = sum(2 * (y_pred[i] - yi) * x[i] for i, yi in enumerate(y)) / len(y)
        grad_b = sum(2 * (y_pred[i] - yi) * 1 for i, yi in enumerate(y)) / len(y)

        # 利用梯度更新参数,并保存每次k和b的值
        k_rec[iteration] = k
        b_rec[iteration] = b
        k -= a * grad_k
        b -= a * grad_b

    if iteration == iterations:
        k_rec[iteration] = (float)(k.detach())
        b_rec[iteration] = (float)(b.detach())

    return k, b


# 产生训练数据，k和b不宜过大或者过小
k = 1
b = 3
num = 50
iterations=2000
k_rec = [0] * iterations
b_rec = [0] * iterations
x_data, y_data = data(num, k, b)
print('真值k:', k)
print('真值b:', b)

# 利用梯度下降算法拟合k和b的值
k_est, b_est = gradient_des(x_data, y_data,k_rec,b_rec,iterations)

# # 输出拟合过程图，横坐标迭代次数，纵坐标k和b的值,红色为k，绿色为b
# plt.plot(range(0, iterations), k_rec, color='red', lw=2, marker='o', linestyle='--')
# plt.plot(range(0, iterations), b_rec, color='green', lw=2, marker='o', linestyle='--')
# plt.xlabel('迭代次数(绿色为b，红色为=k)'), plt.ylabel('k和b的预估值'), plt.xlim(0, iterations * 1.2), plt.ylim(0,5), plt.show()
# print('预估的k:', k_est)
# print('预估的b:', b_est)
