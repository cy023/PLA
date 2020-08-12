# -*- coding: utf-8 -*-
'''
@author : cy023
@date   : 2020.8.9
@brief  : Dataset Visualization.
'''
import matplotlib.pyplot as plt
import numpy as np

def read_txt_data(data_path):
    data = np.loadtxt(data_path)
    x = data[:, :-1]
    x = np.insert(x, 0, 1, axis=1)
    y = data[:, -1:]
    y = np.where(y == 1, -1, 1)
    return np.array(list(zip(x, y)), dtype=object)

def plot_pla(dataset):
    P = [vector[0] for vector in dataset]
    
    C1_area = [i for i in range(dataset.shape[0]) if dataset[i][1] ==  1]
    C2_area = [i for i in range(dataset.shape[0]) if dataset[i][1] == -1]

    plt.scatter([P[i][1] for i in C1_area], [P[i][2] for i in C1_area], s=10, c='b', marker="o", label='Class1')
    plt.scatter([P[i][1] for i in C2_area], [P[i][2] for i in C2_area], s=10, c='r', marker="o", label='Class2')
    plt.tight_layout()
    plt.legend()
    plt.show()



if __name__ == "__main__":

    f = open("data_filename.txt", 'r')
    files = f.readlines()
    f.close()

    for i in range(len(files)):
        files[i] = files[i].strip('\n')
        dataset = read_txt_data("./data/" + str(files[i]))
        plot_pla(dataset)
