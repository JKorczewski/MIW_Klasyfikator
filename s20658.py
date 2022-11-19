import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from perceptron import Perceptron
from plotka import plot_decision_regions
from reglog import LogisticRegressionGD


class Classifier:
    def __init__(self, ppn1, ppn2, ppn3):
        self.ppn1 = ppn1
        self.ppn2 = ppn2
        self.ppn3 = ppn3

    def predict(self, x):
        return np.where(self.ppn1.predict(x) == 1, 0, np.where(self.ppn2.predict(x) == 1, 1, np.where(self.ppn2.predict(x) == 1, 2)))


def main():
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

    y_train_01_subset = y_train.copy()
    y_train_02_subset = y_train.copy()
    y_train_03_subset = y_train.copy()

    y_train_01_subset[y_train != 2] = -1
    y_train_01_subset[y_train == 2] = 1

    y_train_02_subset[y_train != 1] = -1
    y_train_02_subset[y_train == 1] = 1

    y_train_03_subset[y_train != 0] = -1
    y_train_03_subset[y_train == 0] = 1

    # Models learn
    ppn1 = Perceptron(eta=0.1, n_iter=300)
    ppn1.fit(x_train, y_train_01_subset)

    ppn2 = Perceptron(eta=0.1, n_iter=300)
    ppn2.fit(x_train, y_train_02_subset)

    ppn3 = Perceptron(eta=0.1, n_iter=300)
    ppn3.fit(x_train, y_train_03_subset)

    y_train_01_subset_lr = y_train.copy()
    y_train_02_subset_lr = y_train.copy()
    y_train_03_subset_lr = y_train.copy()

    y_train_01_subset_lr[y_train != 2] = -1
    y_train_01_subset_lr[y_train == 2] = 1
    y_train_01_subset_lr[y_train == -1] = 0

    y_train_02_subset_lr[y_train != 1] = -1
    y_train_02_subset_lr[y_train == 1] = 1
    y_train_02_subset_lr[y_train == -1] = 0

    y_train_03_subset_lr[y_train != 0] = -1
    y_train_03_subset_lr[y_train == 0] = 1
    y_train_03_subset_lr[y_train == -1] = 0

    lrGD1 = LogisticRegressionGD(eta=0.1, n_iter=1000, random_state=3)
    lrGD2 = LogisticRegressionGD(eta=0.1, n_iter=1000, random_state=1)
    lrGD3 = LogisticRegressionGD(eta=0.1, n_iter=1000, random_state=1)

    lrGD1.fit(x_train, y_train_01_subset_lr)
    lrGD2.fit(x_train, y_train_02_subset_lr)
    lrGD3.fit(x_train, y_train_03_subset_lr)

    y1_predict = ppn1.predict(x_train)
    y2_predict = ppn2.predict(x_train)
    y3_predict = ppn2.predict(x_train)

    accuracy_1 = accuracy(ppn1.predict(x_train), y_train_01_subset)
    accuracy_2 = accuracy(ppn2.predict(x_train), y_train_02_subset)
    accuracy_3 = accuracy(ppn3.predict(x_train), y_train_03_subset)
    print("Perceptron #1 accuracy: ", accuracy_1)
    print("Perceptron #2 accuracy: ", accuracy_2)
    print("Perceptron #3 accuracy: ", accuracy_3)

    # Calculating accuracy for the whole set
    if accuracy_1 > accuracy_3:
        y_results = np.where(y1_predict == 0, 0, np.where(y3_predict == 1, 2, 1))
    else:
        y_results = np.where(y3_predict == 0, 2, np.where(y1_predict == 1, 0, 1))

    print("Total accuracy: ", accuracy(y_results, y_train))

    _classifier = Classifier(lrGD1, lrGD2, lrGD3)

    # make graphs
    plt.subplot(1, 2, 1)
    plot_decision_regions(x=x_test, y=y_test, classifier=_classifier)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.title("PERCEPTRON")

    plt.subplot(1, 2, 2)
    plot_decision_regions(x=x_test, y=y_test, classifier=_classifier)
    plt.xlabel('Petal length')
    plt.legend(loc='upper left')
    plt.title('LRGD')
    plt.show()


def accuracy(y_results, y_train):
    return (1 - np.mean(y_results != y_train)) * 100


if __name__ == '__main__':
    main()
