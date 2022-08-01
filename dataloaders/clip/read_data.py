import pandas
import numpy
import os

def read_different_file(path):
    different = []
    for dirpath, dirnames, filenames in os.walk(path):
        filenames_len = filenames.__len__()
        if filenames_len:
            for filename in filenames:
                a = filename.split('.')[0].replace('_15label', '')
                if a  not in different:
                    different.append(a)

    return different

def read_different_file_path(files, path):
    train = []
    val = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames_len = dirnames.__len__()
        if dirnames_len:
            for dirname in dirnames:
                flag = 0
                for file in files:
                    if file in dirname:
                        train.append(dirname)
                        flag = 1
                if flag == 0:
                    val.append(dirname)

    print(train, val)
    return train, val

if __name__ == '__main__':
    path = '../../../data/gid-15/GID/ann_dir/train/'
    file = read_different_file(path)
    f = open("../../data/lists/gid15_train.txt", 'w')
    for line in file:
        f.write(line + '\n')
    f.close()
    # path = '/media/dell/DATA/wy/data/GID-15/train_image/scratch/kpyang/GID/origin'
    # file = read_different_file(path)
    # dir_path = '/media/dell/DATA/wy/data/512/image'
    # train, val = read_different_file_path(file, dir_path)
    # f = open("../../data/lists/hps_train.txt", 'w')
    # for line in train:
    #     f.write(line + '\n')
    # f.close()
    #
    # f = open("../../data/lists/hps_val.txt", 'w')
    # for line in val:
    #     f.write(line + '\n')
    # f.close()


