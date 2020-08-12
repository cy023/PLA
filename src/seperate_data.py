# -*- coding: utf-8 -*-
'''
@author : cy023
@date   : 2020.8.12
@brief  : Seperated Dataset Visualization.
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

def plot_data(training, testing):
    P_train = [vector[0] for vector in training]
    P_test  = [vector[0] for vector in testing]
    
    C1_train = [i for i in range(training.shape[0]) if training[i][1] ==  1]
    C2_train = [i for i in range(training.shape[0]) if training[i][1] == -1]

    C1_test  = [i for i in range(testing.shape[0]) if testing[i][1] ==  1]
    C2_test  = [i for i in range(testing.shape[0]) if testing[i][1] == -1]

    plt.scatter([P_train[i][1] for i in C1_train], [P_train[i][2] for i in C1_train], s=10, c='b', marker="o", label='O')
    plt.scatter([P_train[i][1] for i in C2_train], [P_train[i][2] for i in C2_train], s=10, c='r', marker="x", label='X')

    plt.scatter([P_test[i][1] for i in C1_test], [P_test[i][2] for i in C1_test], s=10, c='black', marker="o", label='O')
    plt.scatter([P_test[i][1] for i in C2_test], [P_test[i][2] for i in C2_test], s=10, c='black', marker="x", label='X')

    plt.show()


if __name__ == "__main__":

    dataset = read_txt_data('./data/2CS.txt')
    np.random.shuffle(dataset)

    sep = int((2/3)*dataset.shape[0])

    training_data = dataset[:sep]
    testing_data  = dataset[sep:]
    
    plot_data(training_data, testing_data)
