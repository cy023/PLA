import matplotlib.pyplot as plt
import numpy as np

def sign(x):
    return np.sign(x)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    return np.where(x<0, 0, x)

def prelu(x):
    return np.where(x<0, 0.5*x, x)
 
 
if __name__ == "__main__":

    x = np.arange(-10, 10, 0.1)
    
    # Figure 1
    fig1 = plt.figure()
    
    ## sign
    y = sign(x)
    plt.plot(x, y)
    plt.grid()
    plt.title('sign')

    fig1.tight_layout()
    plt.savefig("./images/sign_function.png")
    plt.show()

    # Figure 2
    fig2 = plt.figure()

    ## sigmoid
    ax1 = fig2.add_subplot(221)
    y1 = sigmoid(x)
    ax1.plot(x, y1)
    ax1.grid()
    ax1.set_title('sigmoid')

    ## tanh
    ax2 = fig2.add_subplot(222)
    y2 = tanh(x)
    ax2.plot(x, y2)
    ax2.grid()
    ax2.set_title('tanh')

    ## relu
    ax3 = fig2.add_subplot(223)
    y3 = relu(x)
    ax3.plot(x, y3)
    ax3.grid()
    ax3.set_title('relu')

    ## prelu
    ax4 = fig2.add_subplot(224)
    y4 = prelu(x)
    ax4.plot(x, y4)
    ax4.grid()
    ax4.set_title('prelu')

    fig2.tight_layout()
    plt.savefig("./images/activation_function.png")
    plt.show()
