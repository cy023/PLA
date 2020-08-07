# -*- coding: utf-8 -*-
'''
@author : cy023
@date   : 2020.8.6
@brief  : Pocket Algorithm Implement.
        With update every training result presents by figure and save in .gif file.

@Note   : 儲存成 gif 檔的方法是將每回合訓練結果儲存到 image_list 之中。
        訓練結束後，再將 image_list 轉為 .gif 檔。
'''
import matplotlib.pyplot as plt
import numpy as np
import imageio

image_list = []

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
                    update_plot(w, dataset)
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

def shuffle(dataset):
    return np.take(dataset, np.random.permutation(dataset.shape[0]), axis=0)

def plot_data(dataset):
    P = [vector[0] for vector in dataset]
    
    C1_area = [i for i in range(dataset.shape[0]) if dataset[i][1] ==  1]
    C2_area = [i for i in range(dataset.shape[0]) if dataset[i][1] == -1]

    plt.scatter([P[i][1] for i in C1_area], [P[i][2] for i in C1_area], s=10, c='b', marker="o", label='O')
    plt.scatter([P[i][1] for i in C2_area], [P[i][2] for i in C2_area], s=10, c='r', marker="x", label='X')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.legend(loc='upper right')

def plot_w(w):
    x = np.linspace(-10, 10)
    a, b = -w[1]/w[2], -w[0]/w[2]
    plt.plot(x, a*x + b, 'b-')
    plt.plot(0, 0, 'g+')

def update_plot(w, dataset):
    plt.clf()
    plot_data(dataset)
    plot_w(w)
    plt.draw()
    plt.savefig('./Images/temp.png')
    image_list.append(imageio.imread('./Images/temp.png'))
    plt.pause(0.2)



if __name__ == "__main__":

    dataset = read_txt_data('./data/2CS.txt')
    dataset = shuffle(dataset)

    eta = float(input('please input learning rate '))

    fig = plt.figure()
    plt.ion()
    
    pla_pocket(eta, dataset)

    # 最後一張圖多重複幾次，有停止的效果，比較好看
    for i in range(5):
        image_list.append(image_list[len(image_list)-1])
    
    imageio.mimsave('./Images/result.gif', image_list, duration=0.2)

    plt.ioff()
    plt.show()
