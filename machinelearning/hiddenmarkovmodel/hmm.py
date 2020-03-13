#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Zhang Kai
# Email : zhangkai7@sgepri.sgcc.com.cn
# Time  : 2020/3/13 9:27
# Function: 隐马尔科夫模型
import pickle


class HMM(object):
    def __init__(self, list_tags, path_model):
        """
        初始化模型
        :param list_tags: 状态集合列表['B', 'I', 'O']
        :param path_model: 训练好的模型位置r'./hmm_model.pkl'
        """
        self.list_tags = list_tags
        self.__path_model = path_model
        self.__dic_start_transitions = dict()  # 初始化初始概率
        self.__dic_transitions = dict()  # 初始化转移概率
        self.__dic_emission = dict()  # 初始化发射概率
        self.__dic_count = dict()
        self.__set_words = None
        self.__flag_load = False

    def __reset_parameter(self):
        """初始化模型参数"""
        for tag in self.list_tags:
            self.__dic_count[tag] = 0
            self.__dic_transitions[tag] = {t: 0.0 for t in self.list_tags}
            self.__dic_start_transitions[tag] = 0.0
            self.__dic_emission[tag] = {w: 0.0 for w in self.__set_words}

    def load_model(self):
        """读取模型"""
        with open(self.__path_model, 'rb') as f:
            self.__dic_start_transitions = pickle.load(f)
            self.__dic_transitions = pickle.load(f)
            self.__dic_emission = pickle.load(f)
            self.__set_words = pickle.load(f)
            self.__flag_load = True

    def train(self, list_text, list_tags):
        """
        训练转移概率、发射概率、初始概率
        :param list_text: 语料库字典
        :param list_tags: 语料库标注
        :return: 保存模型到路径
        """
        assert len(list_text) == len(list_tags)

        #  初始化训练前状态
        self.__set_words = set()  # 初始化字符集合
        for line in list_text:
            self.__set_words |= set(line)
        self.__reset_parameter()

        #  读取语料库内容统计
        num_line = len(list_tags)
        for n in range(num_line):
            line_word = list_text[n]
            line_tags = list_tags[n]
            self.__dic_start_transitions[line_tags[0]] += 1
            for i in range(len(line_tags)):
                tag = line_tags[i]
                word = line_word[i]
                if i == 0:
                    self.__dic_start_transitions[tag] += 1
                else:
                    self.__dic_transitions[line_tags[i - 1]][tag] += 1
                self.__dic_emission[tag][word] += 1
                self.__dic_count[tag] += 1
        #  概率计算
        self.__dic_start_transitions = {x: y * 1.0 / num_line for x, y in self.__dic_start_transitions.items()}
        self.__dic_transitions = {x: {m: n * 1.0 / self.__dic_count[x] for m, n in y.items()} for x, y in
                                  self.__dic_transitions.items()}
        self.__dic_emission = {x: {m: n / self.__dic_count[x] for m, n in y.items()} for x, y in
                               self.__dic_emission.items()}

        #  持久化保存
        with open(self.__path_model, 'wb') as f:
            pickle.dump(self.__dic_start_transitions, f)
            pickle.dump(self.__dic_transitions, f)
            pickle.dump(self.__dic_emission, f)
            pickle.dump(self.__set_words, f)
        self.__flag_load = True

    def __viterbi(self, text):
        prob_history = [dict()]
        path_history = dict()
        for tag in self.list_tags:
            prob_history[0][tag] = self.__dic_start_transitions[tag] * self.__dic_emission[tag].get(text[0], 0)
            path_history[tag] = [tag]
        for i in range(1, len(text)):
            prob_history.append(dict())
            path_next = dict()
            flag_unk = text[i] not in self.__set_words
            for tag in self.list_tags:
                prob_emission = self.__dic_emission[tag].get(text[i], 0) if not flag_unk else 1.0
                prob_best, tag_best = max(
                    [(prob_history[i - 1][t] * self.__dic_transitions[t].get(tag, 0) * prob_emission, t)
                     for t in self.list_tags if prob_history[i - 1][t] > 0])
                prob_history[i][tag] = prob_best
                path_next[tag] = path_history[tag_best] + [tag]
            path_history = path_next
        prob_best, tag_best = max([(prob_history[len(text) - 1][t], t) for t in self.list_tags])
        return prob_best, path_history[tag_best]

    def predict(self, text):
        if not self.__flag_load:
            print('预测前请先训练或读取训练好的模型！')
            return None
        _, pos_list = self.__viterbi(text)
        print(text, '\n', pos_list)
        return pos_list


if __name__ == '__main__':
    """测试"""
    hmm = HMM(['B', 'M', 'S'], r'./db/hmm_model.pkl')
    # text_cut = []
    # text_train = []
    # text_label = []
    # list_tags = ['B', 'M', 'S']
    # with open(r'./db/trainCorpus.txt', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.strip()
    #         if len(line) == 0:
    #             continue
    #         list_line = line.split()
    #         list_label = []
    #         for w in list_line:
    #             length_word = len(w)
    #             if length_word == 0:
    #                 continue
    #             elif length_word == 1:
    #                 list_label.append('S')
    #             else:
    #                 list_label += ['B'] + ['M'] * (length_word - 1)
    #         text_line = ''.join(list_line)
    #         list_text = [x for x in text_line]
    #         if len(list_text) != len(list_label):
    #             print('标注错误')
    #             continue
    #         text_train.append(list_text)
    #         text_label.append(list_label)
    # hmm.train(text_train, text_label)
    hmm.load_model()
    hmm.predict('结婚的和尚未结婚的。')
