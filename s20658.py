import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from perceptron import Perceptron
from plotka import plot_decision_regions
from reglog import LogisticRegressionGD


class Classifier:
    """
    Klasyfikator dla perceptronów. Zawiera metode predykcji.
    :param lrgd1: regresja logiczna dla 2
    :param lrgd2: regresja logiczna dla 1
    :param lrgd3: regresja logiczna dla 0
    :return: None
    """
    def __init__(self, ppn1, ppn2, ppn3):
        self.ppn1 = ppn1
        self.ppn2 = ppn2
        self.ppn3 = ppn3

    def predict(self, x):
        return np.where(self.ppn1.predict(x) == 1, 2, np.where(self.ppn2.predict(x) == 1, 1,
                                                               np.where(self.ppn3.predict(x) == 1, 0, 2)))


class LRGDClassifier:
    """
    Klasyfikator dla regresji logicznej. Zawiera metody predykcji i prawdopodobieństwa.
    :param lrgd1: regresja logiczna dla 2
    :param lrgd2: regresja logiczna dla 1
    :param lrgd3: regresja logiczna dla 0
    :return: None
    """
    def __init__(self, lrgd1, lrgd2, lrgd3):
        self.lrgd1 = lrgd1
        self.lrgd2 = lrgd2
        self.lrgd3 = lrgd3

    def predict(self, x):
        return np.where(self.lrgd1.predict(x) == 1, 2, np.where(self.lrgd2.predict(x) == 1, 1,
                                                                np.where(self.lrgd3.predict(x) == 1, 0, 2)))

    def probability(self, x):
        print(f'Class 0: {round(self.lrgd1.probability_of_reg(x), 6) * 100}%')
        print(f'Class 1: {round(self.lrgd2.probability_of_reg(x), 6) * 100}%')
        print(f'Class 2: {round(self.lrgd3.probability_of_reg(x), 6) * 100}%')


def main():
    """
    Funkcja ładuje zbiór danych z iris, trenuje perceptrony oraz regresje logiczną.
    Po wszystkim funkcja wypisuje prawdopodobieństwo oraz grafy.
    :return: None
    """
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

    # Kopiowanie zasobów w celu stworzenia perceptronów
    y_train_01_subset = y_train.copy()
    y_train_02_subset = y_train.copy()
    y_train_03_subset = y_train.copy()

    # Przystosowanie danych do perceptronów
    y_train_01_subset[y_train != 2] = -1
    y_train_01_subset[y_train == 2] = 1

    y_train_02_subset[y_train != 1] = -1
    y_train_02_subset[y_train == 1] = 1

    y_train_03_subset[y_train != 0] = -1
    y_train_03_subset[y_train == 0] = 1

    # Stworzenie perceptronów
    ppn1 = Perceptron(eta=0.1, n_iter=300)
    ppn1.fit(x_train, y_train_01_subset)

    ppn2 = Perceptron(eta=0.1, n_iter=300)
    ppn2.fit(x_train, y_train_02_subset)

    ppn3 = Perceptron(eta=0.1, n_iter=300)
    ppn3.fit(x_train, y_train_03_subset)

    # Stworzenie klasyfikatora
    _classifierPPN = Classifier(ppn1, ppn2, ppn3)

    # Skopiowanie danych do regresji logicznej
    y_train_01_subset_lr = y_train.copy()
    y_train_02_subset_lr = y_train.copy()
    y_train_03_subset_lr = y_train.copy()

    # Przystosowanie danych do regresji logicznej
    y_train_01_subset_lr[y_train_01_subset_lr != 2] = -1
    y_train_01_subset_lr[y_train_01_subset_lr == 2] = 1
    y_train_01_subset_lr[y_train_01_subset_lr == -1] = 0

    y_train_02_subset_lr[y_train_02_subset_lr != 1] = -1
    y_train_02_subset_lr[y_train_02_subset_lr == 1] = 1
    y_train_02_subset_lr[y_train_02_subset_lr == -1] = 0

    y_train_03_subset_lr[y_train_03_subset_lr != 0] = -1
    y_train_03_subset_lr[y_train_03_subset_lr == 0] = 1
    y_train_03_subset_lr[y_train_03_subset_lr == -1] = 0

    # Stworzenie regresji logicznej
    lrGD1 = LogisticRegressionGD(eta=0.1, n_iter=300, random_state=1)
    lrGD2 = LogisticRegressionGD(eta=0.1, n_iter=300, random_state=1)
    lrGD3 = LogisticRegressionGD(eta=0.1, n_iter=300, random_state=3)

    lrGD1.fit(x_train, y_train_01_subset_lr)
    lrGD2.fit(x_train, y_train_02_subset_lr)
    lrGD3.fit(x_train, y_train_03_subset_lr)

    # Stworzenie klasyfikatora dla regresji logicznej
    _classifierLR = LRGDClassifier(lrGD1, lrGD2, lrGD3)

    # Wypisanie prawdopodobieństwa
    for i in range(x_test.shape[0]):
        print(f'Sample: {x_test[i]}, Real class: {y_test[i]}')
        _classifierLR.probability(x_test[i])

    # Rysowanie grafu
    plt.subplot(1, 2, 1)
    plot_decision_regions(x=x_test, y=y_test, classifier=_classifierPPN)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.title("PERCEPTRON")

    plt.subplot(1, 2, 2)
    plot_decision_regions(x=x_test, y=y_test, classifier=_classifierLR)
    plt.xlabel('Petal length')
    plt.legend(loc='upper left')
    plt.title('LRGD')
    plt.show()


if __name__ == '__main__':
    main()
