import numpy as np
import sys
import argparse
sys.path.append("../")
from model.decode.first_decoder import get_first_space
from model.decode.second_decoder import get_second_space
from model.decode.third_decoder import get_retrain_space

def obtain_encode_args():
    parser = argparse.ArgumentParser(description="get model encode config")

    parser.add_argument('--stage', type=str, default=None, choices=['first', 'second', 'third', 'retrain'], help='model stage')
    parser.add_argument('--layers', type=int, default=12, help='supernet layers number')
    parser.add_argument('--save_path', type=str, default=None, help='model encode save path')
    parser.add_argument('--model_core_path', type=str, default=None, help='model encode core path save path')
    parser.add_argument('--betas_path_stage1', type=str, default=None,  help='stage1 arch_parameters save path')
    parser.add_argument('--betas_path_stage2', type=str, default=None,  help='stage2 arch_parameters save path')
    parser.add_argument('--alphas_path', type=str, default=None,  help='stage3 cell_arch_parameters save path')
    args = parser.parse_args()
    return args
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


def second_connect(layer, path):
    connections = []
    for i in range(layer):
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

def test_connections(connections):
    layers = [connection[1][0] for connection in connections]
    layers.sort()
    for connection in connections:
        if connections.count(connection) != 1:
            print("have samed connections")
            exit()
    return True
if __name__ == "__main__":
    args = obtain_encode_args()
    if args.stage == 'first':
        # first connections
        connections = first_connect_4(args.layers - 1) #
        np.save(args.save_path, connections)

    elif args.stage == 'second':
        # second connections
        betas_path = args.betas_path_stage1
        path = get_first_space(betas_path)
        np.save(args.model_core_path, path[-1])
        connections = second_connect(args.layers, path[-1])
        if test_connections(connections):
            np.save(args.save_path, connections)

    elif args.stage == 'third':
        # third connections
        betas_path_stage1 = args.betas_path_stage1
        betas_path_stage2 = args.betas_path_stage2
        core_path = get_first_space(betas_path_stage1)[-1]
        used_betas = get_second_space(betas_path_stage2, core_path)[-1]
        connections = third_connect(used_betas)
        if test_connections(connections):
            np.save(args.save_path, connections)
            
    elif args.stage == 'retrain':
        alphas_path = args.alphas_path
        cell_arch = get_retrain_space(alphas_path)[-1]
        np.save(args.save_path, cell_arch)
        

