import fnmatch
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    file = open("C:/Users/Faye/Desktop/nas-segm-pytorch-master/src/data/lists/train+.lst", "r", encoding='UTF-8')
    file_list = file.readlines()  # 将所有变量读入列表file_list1
    file_deal = []
    for i in range(len(file_list)):
        number = []
        line = file_list[i]
        chooice = 0
        for j in range(len(line)):
            if line[j:j+3] == '../':
                number.append(j)
                chooice = 1
        if chooice == 1:
            file_deal.append(line[0:number[0]] + line[number[0]+3:number[1]] + line[number[1]+3:])
        else:
            file_deal.append(file_list[i])

    f2 = open('C:/Users/Faye/Desktop/nas-segm-pytorch-master/data/lists2.lst', 'w')
    f2.writelines(file_deal)
