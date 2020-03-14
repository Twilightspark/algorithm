#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Zhang Kai
# Email : zhangkai7@sgepri.sgcc.com.cn
# Time  : 2020/3/14 9:21
# Function: 条件随机场

import torch
import torch.nn as nn


class ConditionalRandomField(nn.Module):
    """线性条件随机场"""

    def __init__(self, num_tag, reset_range=0.01):
        """
        初始化实例
        :param num_tag: 标签总个数
        :param reset_range: 初始化转移矩阵范围
        """
        super(ConditionalRandomField, self).__init__()
        self.num_tags = num_tag
        self._start_transitions = nn.Parameter(torch.empty(num_tag), requires_grad=True)
        self._transitions = nn.Parameter(torch.empty(num_tag, num_tag), requires_grad=True)
        self._end_transitions = nn.Parameter(torch.empty(num_tag), requires_grad=True)
        self._reset_parameters(reset_range)

    def _reset_parameters(self, reset_range):
        """
        初始化模型转移矩阵参数
        :param reset_range: 初始化转移矩阵范围
        """
        nn.init.uniform_(self._start_transitions, 0., reset_range)
        nn.init.uniform_(self._transitions, 0., reset_range)
        nn.init.uniform_(self._end_transitions, 0., reset_range)

    def forward(self, emissions, tags, masks=None):
        """
        前向传播过程
        :param emissions: 观测概率矩阵(batch, seq_len, num_tag)
        :param tags: 实际标签(batch, seq_len)
        :param masks: 填充标识(batch, seq_len)
        :return: loss
        """
        if masks is None:
            masks = torch.ones_like(tags, dtype=torch.uint8)
        #  计算句子得分
        num_score = self._conpute_score(emissions, tags, masks)
        #  计算标准化得分
        sum_score = self._conpute_normalize(emissions, masks)
        loss_score = sum_score - num_score
        return loss_score.mean()

    def decode(self, emissions, masks=None):
        """
        求解最佳标签
        :param emissions: 观测概率矩阵(batch, seq_len, num_tag)
        :param masks: 填充标识(batch, seq_len)
        :return: 最佳标签列表
        """
        if masks is None:
            masks = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)
        return self._viterbi(emissions, masks)

    def _conpute_score(self, emissions, tags, masks):
        """
        句子标签得分
        :param emissions: 观测概率矩阵(batch, seq_len, num_tag)
        :param tags: 实际标签(batch, seq_len)
        :param masks: 填充标识(batch, seq_len)
        :return: 句子得分
        """
        batch_size, seq_length = tags.shape
        seq_end = masks.long().sum(dim=1) - 1
        masks = masks.float()

        score = self._start_transitions[tags[:, 0]]
        score += emissions[torch.arange(batch_size), 0, tags[:, 0]]
        for i in range(1, seq_length):
            score += self._transitions[tags[:, i - 1], tags[:, i]] * masks[:, i]
            score += emissions[torch.arange(batch_size), i, tags[:, i]] * masks[:, i]

        last_tags = tags[torch.arange(batch_size), seq_end]
        score += self._end_transitions[last_tags]
        return score

    def _conpute_normalize(self, emissions, masks):
        """
        句子所有得分
        :param emissions: 观测概率矩阵(batch, seq_len, num_tag)
        :param masks: 填充标识(batch, seq_len)
        :return: 归一化标签
        """
        seq_length = emissions.shape[1]
        score = self._start_transitions + emissions[:, 0, :]
        for i in range(1, seq_length):
            new_score = score.unsqueeze(2)
            new_emission = emissions[:, i, :].unsqueeze(1)
            next_score = new_score + self._transitions + new_emission
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(masks[:, i].unsqueeze(1), next_score, score)
        score += self._end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi(self, emissions, masks):
        """
        维特比最优路径寻找
        :param emissions: 观测概率矩阵(batch, seq_len, num_tag)
        :param masks: 填充标识(batch, seq_len)
        :return: 最优路径
        """
        batch_size, seq_length = masks.shape
        seq_end = masks.long().sum(dim=1) - 1

        score = self._start_transitions + emissions[:, 0, :]
        his_score = []
        for i in range(1, seq_length):
            new_score = score.unsqueeze(2)
            new_emission = emissions[:, i, :].unsqueeze(1)
            next_score = new_score + self._transitions + new_emission
            next_score, index = next_score.max(dim=1)
            score = torch.where(masks[:, i].unsqueeze(1), next_score, score)
            his_score.append(index)
        score += self._end_transitions

        res_tags = []
        for i in range(batch_size):
            _, end_tag = score[i].max(dim=0)
            best_tags = [end_tag.item()]
            for his in reversed(his_score[:seq_end[i]]):
                next_tag = his[i][best_tags[-1]]
                best_tags.append(next_tag.item())
            best_tags.reverse()
            res_tags.append(best_tags)
        return res_tags


if __name__ == '__main__':
    """测试代码"""
    # crf = ConditionalRandomField(4)
    # emis = torch.tensor([[[0.1, 0.01, 0.001, 2.0001], [0.2, 0.02, 0.002, 2.0002], [0.3, 0.03, 0.003, 2.0003]],
    #                      [[0.5, 0.05, 0.005, 0.0005], [0.7, 0.07, 0.007, 0.0007], [0.9, 0.09, 0.009, 0.0009]]])
    # t = torch.tensor([[1, 2, 3], [0, 3, 1]])
    # m = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.uint8)
    # a = crf(emis, t)
    # a = crf(emis, t, m)
    # b = crf.decode(emis)
    # b = crf.decode(emis, m)
    # print(a, b)
