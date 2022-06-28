import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Hypothesis 1
    Y_pretrained = [0.74,0.58,0.48,0.45,0.45,0.45,0.45,0.44,0.45,0.44,0.45,0.45,0.44,0.44]
    Y_new = [0.77,0.70,0.66,0.6,0.51,0.47,0.42,0.40,0.40,0.40,0.41,0.40,0.40,0.40]
    X = [str(i) for i in range(max(len(Y_pretrained),len(Y_new)))]
    plt.plot(X,Y_pretrained,c='b',label='pre-trained model')
    plt.plot(X, Y_new, c='r', label='initialized randomly model')
    plt.legend()
    plt.title('Hypothesis 1')
    plt.xlabel('Dataset size')
    plt.ylabel('Error rate')
    plt.show()


    # Hypothesis 2
    Y_pretrained = [0.62,0.56,0.50,0.47,0.46,0.46,0.46,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45]
    Y_new = [0.77,0.74,0.68,0.62,0.56,0.52,0.48,0.48,0.48,0.48,0.46,0.46,0.44,0.41,0.4,0.4,0.4,0.4,0.4]
    X = [str(i) for i in range(max(len(Y_pretrained),len(Y_new)))]
    plt.plot(X,Y_pretrained,c='b',label='pre-trained model')
    plt.plot(X, Y_new, c='r', label='initialized randomly model')
    plt.legend()
    plt.title('Hypothesis 2')
    plt.xlabel('Epoch')
    plt.ylabel('Error rate')
    plt.show()