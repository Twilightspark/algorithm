#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Zhang Kai
# Email : zhangkai7@sgepri.sgcc.com.cn
# Time  : 2019/12/7 13:58
# Function: K-近邻算法

from data.normalize.normalize import Normalize


class KNN(object):
    """K近邻算法"""

    def __init__(self, data_set, labels, k=1):
        self.data_set = data_set
        self.labels = labels
        self.k = k

    def fit(self, line):
        """判断某个元素"""
        # 数据归一化-计算距离-排序-计算结论
