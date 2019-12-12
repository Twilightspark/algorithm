#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Zhang Kai
# Email : zhangkai7@sgepri.sgcc.com.cn
# Time  : 2019/12/7 13:58
# Function: K-近邻算法

import copy
import numpy as np


class KNN(object):
    """K近邻算法"""

    def __init__(self, k=1):
        self.k = k

    def __nearest_neighbor(self, labels, distances):
        """判断最近的标签"""
        distances_sort = copy.copy(distances)
        distances_sort.sort()
        index_dic = {}
        for i in range(0, self.k):
            index_min = distances.index(distances_sort[i])
            if labels[index_min] not in index_dic.keys():
                index_dic[labels[index_min]] = 1
            else:
                index_dic[labels[index_min]] += 1
        label_res = max(index_dic, key=index_dic.get)
        return label_res, index_dic[label_res]

    def fit_euclidean(self, sample, data_set, labels):
        """判断某个元素"""
        distances = []
        for i in range(data_set.shape[0]):
            distances.append(((data_set[i, :] - sample) ** 2).sum())  # 计算欧氏距离
        label_res, times = self.__nearest_neighbor(labels, distances)
        return label_res, times / self.k


# if __name__ == "__main__":
#     """测试K近邻法"""
#     data = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 4], [10, 11, 12]])
#     label = [0, 0, 1, 1]
#     a = KNN(2)
#     print(a.fit_euclidean([2, 3, 4], data, label))
