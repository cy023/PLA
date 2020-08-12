# -*- coding: utf-8 -*-
'''
@author : cy023
@date   : 2020.8.12
@brief  : HW1
'''
import matplotlib.pyplot as plt
import numpy as np

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

def pla(eta, dataset, limit=5000):
    w = np.zeros(len(dataset[0][0]))
    count = 0
    while check_error(w, dataset) != 0:
        for x, y in dataset:
            if predict(x, w) != y:
                w += y * eta * np.array(x)
                count += 1
        if count >= limit:
            break
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
    if w[2] == 0:
        a, b = np.sign(w[1])*-65535, np.sign(w[0])*-65535
    else:
        a, b = -w[1]/w[2], -w[0]/w[2]

    P = [vector[0] for vector in dataset]
    C1_area = [i for i in range(dataset.shape[0]) if dataset[i][1] ==  1]
    C2_area = [i for i in range(dataset.shape[0]) if dataset[i][1] == -1]

    plt.figure()
    plt.scatter([P[i][1] for i in C1_area], [P[i][2] for i in C1_area], s=10, c='b', label='Class1')
    plt.scatter([P[i][1] for i in C2_area], [P[i][2] for i in C2_area], s=10, c='r', label='Class2')
    plt.plot(x, a*x + b, 'b-')
    plt.plot(0, 0, 'g+')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    # plt.tight_layout()
    plt.legend()
    plt.show()

def plot_data(training, testing):
    P_train = [vector[0] for vector in training]
    P_test  = [vector[0] for vector in testing]
    
    C1_train = [i for i in range(training.shape[0]) if training[i][1] ==  1]
    C2_train = [i for i in range(training.shape[0]) if training[i][1] == -1]
    C1_test  = [i for i in range(testing.shape[0]) if testing[i][1] ==  1]
    C2_test  = [i for i in range(testing.shape[0]) if testing[i][1] == -1]

    plt.figure()
    plt.scatter([P_train[i][1] for i in C1_train], [P_train[i][2] for i in C1_train], s=10, c='b', marker="o", label='Class1 training data')
    plt.scatter([P_train[i][1] for i in C2_train], [P_train[i][2] for i in C2_train], s=10, c='r', marker="o", label='Class2 training data')
    plt.scatter([P_test[i][1] for i in C1_test], [P_test[i][2] for i in C1_test], s=10, c='black', marker="+", label='Class1 testing data')
    plt.scatter([P_test[i][1] for i in C2_test], [P_test[i][2] for i in C2_test], s=10, c='black', marker="x", label='Class2 testing data')

    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":

    # 讀取 dataset
    # dataset = read_txt_data('./data/perceptron1.txt')
    # dataset = read_txt_data('./data/perceptron2.txt')
    # dataset = read_txt_data('./data/2Ccircle1.txt')
    # dataset = read_txt_data('./data/2Circle1.txt')
    # dataset = read_txt_data('./data/2Circle2.txt')
    # dataset = read_txt_data('./data/2CloseS.txt')
    # dataset = read_txt_data('./data/2CloseS2.txt')
    # dataset = read_txt_data('./data/2CloseS3.txt')
    # dataset = read_txt_data('./data/2cring.txt')
    dataset = read_txt_data('./data/2CS.txt')
    # dataset = read_txt_data('./data/2Hcircle1.txt')
    # dataset = read_txt_data('./data/2ring.txt')

    # 隨機打亂 dataset
    np.random.shuffle(dataset)

    # 切分出 training data, testing data
    sep = int((2/3)*dataset.shape[0])
    training_data = dataset[:sep]
    testing_data  = dataset[sep:]
    # plot_data(training_data, testing_data)

    # 使用者設定 "學習率" 、 "收斂條件"
    eta = float(input('please input learning rate '))
    limit = float(input('please input limit '))

    # 帶入 training data 訓練出回歸線 (w)
    w = pla(eta, training_data, limit)
    print("鍵結值 : ", w)
    plot_pla(w, training_data)
    # plot_pla(w, dataset)

    # 計算測試辨識率
    error = check_error(w, testing_data)
    print("測試資料總數 : {}, 錯誤數量 : {}".format(len(testing_data), error))
    print("測試辨識率 : {:.2%}".format((len(testing_data) - error)/len(testing_data)))