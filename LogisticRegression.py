# Logistic regression from scratch:
from numpy import log, dot, e
from numpy.random import rand
import numpy as np

class LogisticRegression:

    def __init__(self, lr = 0.01, epochs = 11, batch_size = 1):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def sigmoid_func(self, z):
        return 1 / (1 + np.exp(-z))

    def logistic_loss(self, X, y, weights):
        N = len(y)
        sum = 0
        for n in range(N):
            t = dot(weights.transpose(), X[n])
            x = dot(-y[n], t)
            exp = np.exp(x)
            ln = np.log(1 + exp)
            sum += ln

        return sum/N

    def fit(self, X, y):
        loss = []
        # Initial weights
        weights = rand(X.shape[1])
        m = len(y)
        lr = self.lr
        epochs = self.epochs
        threshold = 0.5
        data = X.copy()
        labels = y.copy()

        # Need to use stochastic gradient descent
        for e in range(epochs):
            print('start epoch,', e)
            # Decrease learning rate.
            if e > 5:
                lr *= 0.5
            if e > 7:
                lr *= 0.5
            if e > 10:
                lr *= 0.5

            # Shuffle data
            idx = np.random.permutation(len(data))
            data, labels = data[idx], labels[idx]

            for i in range(m):
                z = dot(data[i], weights)
                prediction = self.sigmoid_func(z)

                if prediction > threshold:
                    prediction = 1
                else:
                    prediction = -1

                weights = weights - (lr * data[i] * (prediction - labels[i]))
                loss.append(self.logistic_loss(data, labels, weights))

            print('training loss:', loss[-1])
            print('end epoch,', e)

        self.weights = weights
        self.loss = loss

    def predict(self, X):
        z = dot(X, self.weights)
        z_sigmoid = self.sigmoid_func(z)
        # Using mean as threshold to improve accuracy (instead of 0.5)
        threshold = z_sigmoid.mean()
        return [1 if i > threshold else -1 for i in z_sigmoid]

    def accuracy(self, labels, predictions):
        N = len(labels)
        # Find correct label count
        correct_count = 0
        for i in range (N):
            if labels[i] == predictions[i]:
                correct_count += 1

        return correct_count/N

def main():

    print("logistic regression from scratch")

    # Training
    train_features = np.load('train_features.npy')
    train_labels = np.load('train_labels.npy')

    # Testing
    test_features = np.load('test_features.npy')
    test_labels = np.load('test_labels.npy')

    lr = LogisticRegression()
    lr.fit(train_features, train_labels)

    print('final weights:', lr.weights)

    predictions = lr.predict(test_features)
    print('Test labels:', test_labels)
    print('My predictions:', predictions)

    accuracy = lr.accuracy(test_labels,predictions)
    print('Accuracy for test data:', accuracy)

if __name__ == '__main__':
    main()
