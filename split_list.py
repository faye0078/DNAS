import pandas as pd
import random
def data_split(full_list, ratio, shuffle=True):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def splitFile():
    file = open("/media/dell/DATA/wy/Seg_NAS/data/lists/gid15_val.lst", "r", encoding='UTF-8')
    file_list = file.readlines()
    mini_data, _ = data_split(file_list, 0.3)
    file_name = []
    for i in range(mini_data.__len__()):
        a = str(file_list[i]).replace('\n', '')
        file_name.append(a)
    df = pd.DataFrame(file_name, columns=['one'])
    df.to_csv('/media/dell/DATA/wy/Seg_NAS/data/lists/mini_gid15_val.lst', columns=['one'], index=False, header=False)

    file.close()
splitFile()