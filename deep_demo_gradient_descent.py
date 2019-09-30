import numpy as np
import matplotlib.pyplot as plt


def init_variables():
    """
        Init model variables (weights, bias)
    """
    weights = np.random.normal(size=2)
    bias = 0

    return weights, bias


def get_dataset():
    """
        Method used to generate the dataset
    """

    # Numbers of row per class
    row_per_class = 100
    # Generate rows
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    sick2 = np.random.randn(row_per_class, 2) + np.array([2, 2])

    healthy = np.random.randn(row_per_class, 2) + np.array([-2, 2])
    healthy2 = np.random.randn(row_per_class, 2) + np.array([2, -2])

    features = np.vstack([sick, sick2, healthy, healthy2])
    targets = np.concatenate(
        (np.zeros(row_per_class * 2), np.zeros(row_per_class * 2) + 1))

    return features, targets


def pre_activation(features, weights, bias):
    """
        Compute pre activation
    """

    return np.dot(features, weights) + bias


def activation(z):
    """
        Compute activation
    """

    return 1/(1+np.exp(-z))


def derivate_activation(z):
    """
    """
    return activation(z) * (1 - activation(z))


def predict(features, weights, bias):
    """
    """

    z = pre_activation(features, weights, bias)
    y = activation(z)
    return np.round(y)


def cost(predictions, targets):
    """
    """
    return np.mean((predictions - targets)**2)


def train(features, targets, weights, bias):
    """
    """
    epochs = 100
    learning_rate = 0.1

    # Print accuracy
    predictions = predict(features, weights, bias)
    print("Accuracy: ", np.mean(predictions == targets))

    # plt.scatter(features[:, 0], features[:, 1], s=40,
    #             c=targets, cmap=plt.cm.Spectral)
    # plt.show()

    for epoch in range(epochs):
        if epoch % 10 == 0:
            predictions = activation(pre_activation(features, weights, bias))
            print("Cost = %s" % cost(predictions, targets))

        # Init gradients
        weights_gradients = np.zeros(weights.shape)
        bias_gradient = 0

        # Go through each row
        for feature, target in zip(features, targets):

            z = pre_activation(feature, weights, bias)
            y = activation(z)
            # update gradients
            weights_gradients += (y - target) * \
                derivate_activation(z) * feature
            bias_gradient += (y - target) * derivate_activation(z)

        # Update variables
        weights = weights - learning_rate * weights_gradients
        bias = bias - learning_rate * bias_gradient

    # Print current Accuracy
    predictions = predict(features, weights, bias)
    print("Accuracy after: ", np.mean(predictions == targets))

    pass


if __name__ == '__main__':
    print("executed...\n")
    # Dataset
    features, targets = get_dataset()
    # Variables
    weights, bias = init_variables()
    train(features, targets, weights, bias)
    pass
