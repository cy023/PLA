# -*- coding: utf-8 -*-
'''
@author : cy023
@date   : 2020.8.3
@brief  : Perceptron Learning Algorithm Implement.
'''
import matplotlib.pyplot as plt
import numpy as np
#TODO : 設定收斂條件

def predict(x, w):
    if w.T.dot(np.array(x)) >= 0:
        return 1
    elif w.T.dot(np.array(x)) < 0:
        return -1

def check_error(w, dataset):
    count_error = 0
    for x, y in dataset:
        if predict(x, w) != y:
            count_error += 1
    return count_error

def pla(eta, dataset):
    w = np.zeros(len(dataset[0][0]))
    count = 0
    while check_error(w, dataset) != 0:
        for x, y in dataset:
            if predict(x, w) != y:
                w += y * eta * np.array(x)
                count += 1
        err_count = check_error(w, dataset)
        if err_count == 0 or count >= 5000:
            break
        print("error count : %d, update count : %d" % (err_count, count))
    return w

def read_txt_data(data_path):
    data = np.loadtxt(data_path)
    x = data[:, :-1]
    x = np.insert(x, 0, 1, axis=1)
    y = data[:, -1:]
    y = np.where(y == 1, -1, 1)
    return np.array(list(zip(x, y)))

def shuffle(dataset):
    return np.take(dataset, np.random.permutation(dataset.shape[0]), axis=0)

def plot_pla(w, dataset):
    x = np.linspace(0, 7.5)
    a, b = -w[1]/w[2], -w[0]/w[2]

    P = [vector[0] for vector in dataset]
    
    C1_area = [i for i in range(dataset.shape[0]) if dataset[i][1] ==  1]
    C2_area = [i for i in range(dataset.shape[0]) if dataset[i][1] == -1]

    plt.scatter([P[i][1] for i in C1_area], [P[i][2] for i in C1_area], s=10, c='b', marker="o", label='O')
    plt.scatter([P[i][1] for i in C2_area], [P[i][2] for i in C2_area], s=10, c='r', marker="x", label='X')
    plt.plot(x, a*x + b, 'b-')
    plt.plot(0, 0, 'g+')
    plt.xlim([0, 7.5])
    plt.ylim([0, 7.5])
    # plt.tight_layout()
    plt.legend()
    plt.savefig("./images/pla.png")
    plt.show()

if __name__ == "__main__":

    dataset = read_txt_data('./data/2CS.txt')
    dataset = shuffle(dataset)

    eta = float(input('please input learning rate '))

    w = pla(eta, dataset)
    plot_pla(w, dataset)
