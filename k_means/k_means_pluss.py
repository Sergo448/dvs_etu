# project name: dvs_etu
# version: 1.0
# file name: kmeans.py
# auther: Sergo448, 7193
# date: 15.11.2021
# Python version : 3.10

import numpy as np


class Make_funks:

    def __init__(self, codevector, ID):
        self.codevector = codevector
        self.sum = [0, 0, 0]
        self.num = 0
        self.ID = ID

    def get_codevector(self):
        return self.codevector

    def get_ID(self):
        return self.ID

    def add_sum(self, add_value):
        for i in range(0, 3):
            self.sum[i] = self.sum[i] + add_value[i]
        self.num = self.num + 1

    def reset_sum(self):
        self.sum = [0, 0, 0]
        self.num = 0

    def set_codevector(self, codevector):
        self.codevector = codevector

    def get_new_codevector(self):
        return [np.ceil(self.sum[0] / self.num), np.ceil(self.sum[1] / self.num), np.ceil(self.sum[2] / self.num)]
