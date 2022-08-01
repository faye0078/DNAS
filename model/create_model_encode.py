import os
import numpy as np
import sys
sys.path.append("../")
from decode.first_decoder import get_first_space
from decode.second_decoder import get_second_space
def first_connect_4(layer):
    connections = []
    connections.append([[-1, 0], [0, 0]])

    for i in range(layer):
        if i == 0:
            connections.append([[0, 0], [1, 0]])
            connections.append([[0, 0], [1, 1]])

        elif i == 1:
            connections.append([[1, 0], [2, 0]])
            connections.append([[1, 0], [2, 1]])
            connections.append([[1, 1], [2, 0]])
            connections.append([[1, 1], [2, 1]])
            connections.append([[1, 1], [2, 2]])

        elif i == 2:
            connections.append([[2, 0], [3, 0]])
            connections.append([[2, 0], [3, 1]])
            connections.append([[2, 1], [3, 0]])
            connections.append([[2, 1], [3, 1]])
            connections.append([[2, 1], [3, 2]])
            connections.append([[2, 2], [3, 1]])
            connections.append([[2, 2], [3, 2]])
            connections.append([[2, 2], [3, 3]])

        else:
            connections.append([[i, 0], [i + 1, 0]])
            connections.append([[i, 0], [i + 1, 1]])
            connections.append([[i, 1], [i + 1, 0]])
            connections.append([[i, 1], [i + 1, 1]])
            connections.append([[i, 1], [i + 1, 2]])
            connections.append([[i, 2], [i + 1, 1]])
            connections.append([[i, 2], [i + 1, 2]])
            connections.append([[i, 2], [i + 1, 3]])
            connections.append([[i, 3], [i + 1, 2]])
            connections.append([[i, 3], [i + 1, 3]])

    return np.array(connections)


def second_connect(layer, depth, path):
    connections = []
    for i in range(layer):
        # connections.append([[i, path[i]], [i + 1, path[i + 1]]])
        for j in range(path[i]+1):
            if j == path[i]:
                connections.append([[-1, 0], [i, j]])
                for k in range(i):
                    for l in range(path[k]+1):
                        connections.append([[k, l], [i, j]])
                continue
            if j == 0:
                if path[i-1] == 0:
                    connections.append([[i - 1, 0], [i, 0]])
                else:
                    connections.append([[i - 1, 0], [i, 0]])
                    connections.append([[i - 1, 1], [i, 0]])

            if j == 1:
                if path[i-1] == 0:
                    connections.append([[i - 1, 0], [i, 1]])
                elif path[i-1] == 1:
                    connections.append([[i - 1, 0], [i, 1]])
                    connections.append([[i - 1, 1], [i, 1]])
                elif path[i-1] == 2:
                    connections.append([[i - 1, 0], [i, 1]])
                    connections.append([[i - 1, 1], [i, 1]])
                    connections.append([[i - 1, 2], [i, 1]])

            if j == 2:
                if path[i-1] == 1:
                    connections.append([[i - 1, 1], [i, 2]])
                elif path[i-1] == 2:
                    connections.append([[i - 1, 1], [i, 2]])
                    connections.append([[i - 1, 2], [i, 2]])
                elif path[i-1] == 3:
                    connections.append([[i - 1, 1], [i, 2]])
                    connections.append([[i - 1, 2], [i, 2]])
                    connections.append([[i - 1, 3], [i, 2]])


    return connections

def third_connect(used_betas):
    connections = []
    core_node = []
    num_add_node = np.zeros(len(used_betas))
    for i in range(len(used_betas)):
        core_node.append([i, len(used_betas[i]) - 1]) # need -1 same to former
        num_add_node[i] = num_add_node[i-1] + len(used_betas[i])
        for j in range(len(used_betas[i])):

            if j == len(used_betas[i]) - 1:
                for k, used_beta in enumerate(used_betas[i][j]):
                    if k == 0:
                        if used_beta == 1:
                            connections.append([[-1, 0], [i, j]])
                    elif k == 1:
                        if used_beta == 1:
                            connections.append([[0, 0], [i, j]])
                    else:
                        for l in range(i):
                            if num_add_node[l] < k <= num_add_node[l + 1]:
                                if used_beta == 1:
                                    connections.append([[l + 1, int(k - num_add_node[l] - 1)], [i, j]])
                if i == 0:
                    connections.append([[-1, 0], [0, 0]])
                else:
                    connections.append([[i - 1, len(used_betas[i - 1]) - 1], [i, j]])

            else:
                if j == 0:
                    if len(used_betas[i][j]) == 1:
                        if used_betas[i][j][0] == 1:
                            connections.append([[i - 1, 0], [i, j]])
                    elif len(used_betas[i][j]) == 2:
                        if used_betas[i][j][0] == 1:
                            connections.append([[i - 1, 0], [i, j]])
                        if used_betas[i][j][1] == 1:
                            connections.append([[i - 1, 1], [i, j]])

                elif j == 1:
                    if len(used_betas[i][j]) == 2:
                        if used_betas[i][j][0] == 1:
                            connections.append([[i - 1, 0], [i, j]])
                        if used_betas[i][j][1] == 1:
                            connections.append([[i - 1, 1], [i, j]])
                    if len(used_betas[i][j]) == 3:
                        if used_betas[i][j][0] == 1:
                            connections.append([[i - 1, 0], [i, j]])
                        if used_betas[i][j][1] == 1:
                            connections.append([[i - 1, 1], [i, j]])
                        if used_betas[i][j][2] == 1:
                            connections.append([[i - 1, 2], [i, j]])

                elif j == 2:
                    if len(used_betas[i][j]) == 2:
                        if used_betas[i][j][0] == 1:
                            connections.append([[i - 1, 1], [i, j]])
                        if used_betas[i][j][1] == 1:
                            connections.append([[i - 1, 2], [i, j]])
                    if len(used_betas[i][j]) == 3:
                        if used_betas[i][j][0] == 1:
                            connections.append([[i - 1, 1], [i, j]])
                        if used_betas[i][j][1] == 1:
                            connections.append([[i - 1, 2], [i, j]])
                        if used_betas[i][j][2] == 1:
                            connections.append([[i - 1, 3], [i, j]])

    return connections

def loop_stage1_cell_arch_init():
    a = np.zeros([14, 4, 12])
    a[:, :, 1] = 1
    return a

# def test_forward(target, connections, layer):
#
#     if target[1][0] == layer:
#         return True
#     forward_connection_list = [connection for connection in connections if connection[0]==target[1]]
#     for forward_connection in forward_connection_list:
#         return test_forward(forward_connection, connections, layer)
#
#
# def test_back(target, connections):
#     if target[0][0] == -1:
#         return True
#     back_connection_list = [connection for connection in connections if connection[1] == target[0]]
#     for back_connection in back_connection_list:
#         return test_forward(back_connection, connections)

def test_connections(connections):
    layers = [connection[1][0] for connection in connections]
    layers.sort()
    for connection in connections:
        if connections.count(connection) != 1:
            print("have samed connections")
            exit()

        # test_forward(connection, connections, layers[-1])
        # if ~test_back(connection, connections) or ~test_forward(connection, connections, layers[-1]):
        #     connections.remove([connection])
    return True
if __name__ == "__main__":
    # first connections
    # connections = first_connect_4(15) #
    # np.save('./model_encode/first_connect_4.npy', connections)  # 保存为.npy格式

    # second connections
    # betas_path = '/media/dell/DATA/wy/Seg_NAS/run/uadataset_dfc/search/16layers_first/experiment_0/betas/'
    # path = get_first_space(betas_path)
    # np.save('./model_encode/core_path.npy', path[59])
    # connections = second_connect(16, 4, path[59])
    # if test_connections(connections):
    #     np.save('./model_encode/second_connect_4.npy', connections)  # 保存为.npy格式

    # third connections
    betas_path_stage1 = '/media/dell/DATA/wy/Seg_NAS/run/uadataset_dfc/search/16layers_first/experiment_0/betas/'
    betas_path_stage2 = '/media/dell/DATA/wy/Seg_NAS/run/uadataset_dfc/search/16layers_second/experiment_0/betas/'
    core_path = get_first_space(betas_path_stage1)[59]
    used_betas = get_second_space(betas_path_stage2, core_path)[59]
    connections = third_connect(used_betas)
    test_connections(connections)
    connections_path = './model_encode/third_connect_4.npy'
    np.save(connections_path, connections)

    # get init cellarch
    # cell_arch = loop_stage1_cell_arch_init()
    # path = './model_encode/uadataset/one_loop_14layers_mixedcell/'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # np.save(path + 'init_cell_arch.npy', cell_arch)
