# -*- coding: UTF-8 -*-
"""
@author: hichenway
@知乎: 海晨威
@contact: lyshello123@163.com
@time: 2020/5/9 17:00
@license: Apache
主程序：包括配置，数据读取，日志记录，绘图，模型训练和预测

@user: sinian_dragon
@重邮: 林哲 2021211322
@use_time: 2024/4/17 10:30
"""
import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from LSTM_pytorch import train, predict  # 导入自定义的LSTM训练和预测函数

class Config:
    def __init__(self):
        # 数据参数
        self.feature_columns = list(range(2, 9))  # 要作为feature的列
        self.label_columns = [4, 5]  # 要预测的列，最低价和最高价
        self.predict_day = 1  # 预测未来几天
        self.label_in_feature_index = [self.feature_columns.index(i) for i in self.label_columns]
        self.add_train = False
        self.do_continue_train = False
        # 网络参数
        self.input_size = len(self.feature_columns)
        self.output_size = len(self.label_columns)
        self.hidden_size = 128
        self.lstm_layers = 2
        self.dropout_rate = 0.2
        self.time_step = 60

        # 训练参数
        self.do_train = True
        self.do_predict = True
        self.shuffle_train_data = True
        self.use_cuda = True
        self.train_data_rate = 0.95
        self.valid_data_rate = 0.25
        self.batch_size = 128
        self.learning_rate = 0.001
        self.epoch = 10
        self.patience = 3
        self.random_seed = 42

        # 路径参数
        self.train_data_path = "cdata.csv"
        self.model_save_path = "./checkpoint/pytorch/"
        self.figure_save_path = "./figure/"
        self.log_save_path = "./log/"
        self.do_log_print_to_screen = True
        self.do_log_save_to_file = True
        self.do_figure_save = False
        self.do_train_visualized = False

        # 设置日志路径
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.figure_save_path):
            os.makedirs(self.figure_save_path)
        if self.do_train and (self.do_log_save_to_file or self.do_train_visualized):
            cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            log_save_path = self.log_save_path + cur_time + '_pytorch'
            os.makedirs(log_save_path)

def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S', fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        log_file_path = os.path.join(config.log_save_path, "out.log")
        file_handler = RotatingFileHandler(log_file_path, maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 记录配置信息到日志文件
        logger.info("Config: {}".format(vars(config)))

    return logger

def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
        data = pd.read_csv(config.train_data_path, usecols=config.feature_columns).values

        # 数据归一化处理
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        norm_data = (data - mean) / std

        train_num = int(norm_data.shape[0] * config.train_data_rate)
        train_data = norm_data[:train_num]
        test_data = norm_data[train_num:]

        train_X, valid_X, train_Y, valid_Y = [], [], [], []
        for i in range(train_num - config.time_step - config.predict_day + 1):
            train_X.append(train_data[i:i + config.time_step])
            train_Y.append(train_data[i + config.time_step:i + config.time_step + config.predict_day])

        for i in range(test_data.shape[0] - config.time_step - config.predict_day + 1):
            valid_X.append(test_data[i:i + config.time_step])
            valid_Y.append(test_data[i + config.time_step:i + config.time_step + config.predict_day])

        train_X, train_Y = np.array(train_X), np.array(train_Y)
        valid_X, valid_Y = np.array(valid_X), np.array(valid_Y)

        if config.do_train:
            assert train_X.shape[0] == train_Y.shape[0]
            assert valid_X.shape[0] == valid_Y.shape[0]
            train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict:
            # 在进行预测时，需要确保模型的输入和输出维度正确匹配
            pred_result = predict(config, valid_X)  # 预测
            # 恢复预测结果的原始尺度
            pred_result = pred_result * std[config.label_in_feature_index] + mean[config.label_in_feature_index]
            draw(config, valid_Y, pred_result, logger)  # 绘制预测结果图表

    except Exception as e:
        logger.error("Run Error: {}".format(str(e)), exc_info=True)


def draw(config, label_data, pred_data, logger):
    # 确保预测结果和真实数据的维度匹配
    assert label_data.shape == pred_data.shape, "Shape mismatch between label_data and pred_data"
    # 计算均方误差
    mse = np.mean((label_data - pred_data) ** 2, axis=0)
    logger.info("Mean Squared Error: {}".format(mse))

    # 绘制预测结果图表
    for i in range(len(config.label_columns)):
        plt.figure()
        plt.plot(label_data[:, i], label='Actual')
        plt.plot(pred_data[:, i], label='Predicted')
        plt.title("Predicted vs Actual for {}".format(config.label_columns[i]))
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main(Config())
