#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Zhang Kai
# Email : zhangkai7@sgepri.sgcc.com.cn
# Time  : 2019/12/7 11:07
# Function: 数据标准化


class Normalize(object):
    """数据标准化类"""

    def __init__(self):
        """初始化"""
        self.max = 0
        self.min = 0
        self.__max = 0
        self.__min = 1

    def __set_max_min(self, line_data):
        """置入最大最小值"""
        try:
            self.max = line_data[0]
            self.min = line_data[0]
        except Exception as e:
            print('发生错误：', e)
        for i in line_data:
            if self.max < i:
                self.max = i
            if self.min > i:
                self.min = i

    def __set_range(self, scope):
        """置入处理后数据的范围"""
        try:
            self.__min = scope[0]
            self.__max = scope[1]
        except Exception as e:
            print('范围设置错误：', e)

    def linear(self, line_data, scope=(0, 1)):
        """普通归一化"""
        self.__set_max_min(line_data)
        self.__set_range(scope)
        for i in range(len(line_data)):
            line_data[i] = (line_data[i] - self.min) / (self.max - self.min) * (self.__max - self.__min) + self.__min
        return line_data

    def single_linear(self, single_data):
        """归一化单个数据"""
        single_data = (single_data - self.min) / (self.max - self.min) * (self.__max - self.__min) + self.__min
        return single_data

    def reduce_linear(self, line_data):
        """普通归一化还原"""
        for i in range(len(line_data)):
            line_data[i] = (line_data[i] - self.__min) / (self.__max - self.__min) * (self.max - self.min) + self.min
        return line_data

if __name__ == "__main__":
    """测试"""
    a = Normalize()
    d = [1444569, 1551972, 1795280, 1716065, 1761956, 2122332, 2262601, 2839882, 3532551, 4708442]
    c = a.linear(d, (0, 1))
    print(c)
    d = a.reduce_linear(c)
    print(d)
