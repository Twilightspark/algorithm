#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Zhang Kai
# Email : zhangkai7@sgepri.sgcc.com.cn
# Time  : 2019/12/11 10:46
# Function: 香农熵 互信息


import math


def shannon_entropy(labels):
    nums = len(labels)
    information_dic = {}
    for i in range(nums):
        label = labels[i]
        if label not in information_dic.keys():
            information_dic[label] = 1
        else:
            information_dic[label] += 1
    entropy = 0
    for key in information_dic:
        p = information_dic[key] / nums
        entropy -= p * math.log(p, 2)
    return entropy

# if __name__=="__main__":
#     labels = ["yes", "yes", "no", "no", "no"]
#     print(shannon_entropy(labels))
#     labels[0] = "maybe"
#     print(shannon_entropy(labels))
