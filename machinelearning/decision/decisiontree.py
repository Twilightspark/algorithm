#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Zhang Kai
# Email : zhangkai7@sgepri.sgcc.com.cn
# Time  : 2019/12/11 11:01
# Function: 决策树算法
import math
import pandas as pd

class DecisionTree(object):
    """决策树算法"""
    def __init__(self):
        """初始化"""
        self.labels = None
        self.data_set = None

    def __shannon_entropy(self):
        """计算香农信息熵"""
        nums = len(self.labels)
        information_dic = {}
        for i in range(nums):
            label = self.labels[i]
            if label not in information_dic.keys():
                information_dic[label] = 1
            else:
                information_dic[label] += 1
        entropy = 0
        for key in information_dic:
            p = information_dic[key] / nums
            entropy -= p * math.log(p, 2)
        return entropy


    def __split_data(self, index, value):
        """分割数据"""


    def iterative_dichotomiser_3(self, information):
        self.labels = information.columns
        self.data_set = information[1:, :]


if __name__ == '__main__':
    information = pd.DataFrame({'头发':['长', '短', '短', '短', '短', '长', '长', '长', '短',],
                                '身高':[170, 175, 180, 160, 165, 160, 155, 175, 165,],
                                '体重':[120, 140, 160, 100, 125, 90, 95, 120, 100,],
                                '性别':['男', '男', '男', '男', '男', '女', '女', '女', '女',]})
    print(information.columns)
    pass
