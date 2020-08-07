# -*- coding: utf-8 -*-
'''
@author : cy023
@date   : 2020.8.6
@brief  : Pocket Algorithm Implement.
'''
import matplotlib.pyplot as plt
import numpy as np

def predict(x, w):
    """
    sign function.
    """
    if w.T.dot(np.array(x)) >= 0:
        return 1
    elif w.T.dot(np.array(x)) < 0:
        return -1

def check_error(w, dataset):
    """
    檢查訓練出來的權重帶入 training data 還有幾筆輸出與 label 不一樣。 
    """
    count_error = 0
    for x, y in dataset:
        if predict(x, w) != y:
            count_error += 1
    return count_error

def pla_pocket(eta, dataset, limit=5000):
    w     = np.zeros(len(dataset[0][0])) # w : 原來的權重
    w_tmp = np.zeros(len(dataset[0][0])) # w_tmp : 修正後的權重
    min_err = 1000000 # 用來存當前算出最準權重的錯誤數量 ，初始值 "無限大"
    count = 0
    while check_error(w, dataset) != 0:
        for x, y in dataset:
            if predict(x, w) != y:
                w_tmp += y * eta * np.array(x)
                err_count = check_error(w_tmp, dataset)
                # 檢查更新後的 w_tmp 有沒有比較準，如果有比較準就用，如果沒比較準就用原來的。
                if err_count <= min_err:
                    min_err = err_count
                    w = w_tmp
                count += 1
        if count >= limit:
            break
        print("error count : %d, update count : %d" % (check_error(w, dataset), count))
    return w

def read_txt_data(data_path):
    data = np.loadtxt(data_path)
    x = data[:, :-1]
    x = np.insert(x, 0, 1, axis=1)
    y = data[:, -1:]
    y = np.where(y == 1, -1, 1)
    return np.array(list(zip(x, y)), dtype=object)

def plot_pla(w, dataset):
    x = np.linspace(-10, 10)
    a, b = -w[1]/w[2], -w[0]/w[2]

    P = [vector[0] for vector in dataset]
    
    C1_area = [i for i in range(dataset.shape[0]) if dataset[i][1] ==  1]
    C2_area = [i for i in range(dataset.shape[0]) if dataset[i][1] == -1]

    plt.scatter([P[i][1] for i in C1_area], [P[i][2] for i in C1_area], s=10, c='b', marker="o", label='O')
    plt.scatter([P[i][1] for i in C2_area], [P[i][2] for i in C2_area], s=10, c='r', marker="x", label='X')
    plt.plot(x, a*x + b, 'b-')
    plt.plot(0, 0, 'g+')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    # plt.tight_layout()
    plt.legend()
    plt.savefig("./images/pla_pocket.png")
    plt.show()



if __name__ == "__main__":

    dataset = read_txt_data('./data/2Ccircle1.txt')
    np.random.shuffle(dataset)

    eta = float(input('please input learning rate '))

    w = pla_pocket(eta, dataset)
    plot_pla(w, dataset)
