import numpy as np
import tqdm


class SOM(object):
    def __init__(self, centers, lr, decay_steps, input_shape=None, verbose=True):
        """
        SOM with Winner take all approach
        :param centers: number of centroids to find
        :param lr: initial learning rate
        :param decay_steps: learning rate decay steps
        :param verbose: Verbose output
        :param input_shape: tuple with (# inputs, # patterns)
        """
        self.centers = centers
        self.decay_steps = decay_steps
        self.lr = lr
        self.verbose = verbose
        self.kernel = None

        if input_shape is not None:
            self.inputs = input_shape[0]
            self.patterns = input_shape[1]
            self.built = True
        else:
            self.inputs = None
            self.patterns = None
            self.built = False

    def _build(self, input_shape):
        self.inputs = input_shape[0]
        self.patterns = input_shape[1]
        self.built = True

    def fit(self, x, epochs):
        """
        Fit the SOM model
        :param x: input parameter
        :param epochs: number of epoch to train
        """
        if not self.built:
            self._build(x.shape[1:])

        ep_iterator = tqdm.trange(epochs, disable=self.verbose)
        R = np.random.permutation(self.patterns)

        self.kernel = x[:, R[:self.centers]]

        for k in ep_iterator:
            for i in tqdm.trange(self.patterns, disable=self.verbose):
                lr = self.lr * np.exp(-k / self.decay_steps)
                x_c = np.broadcast_to(x[:, i], (self.centers, 2)).T
                loss = np.linalg.norm(x_c - self.kernel, axis=0)
                mx, pos = np.amin(loss), np.argmin(loss)
                dW = lr * (x[:, i] - self.kernel[:, pos])
                self.kernel[:, pos] = self.kernel[:, pos] + dW

    def evaluate(self, x):
        """
        :param x: input to evaluate
        :return: Clusters with evaluation of x
        """
        return self.__call__(x)

    def __call__(self, x):
        """
        :param x: input to evaluate
        :return: Clusters with evaluation of x
        """

        if self.kernel is None:
            raise RuntimeError('Weights are not initialized, please use .fit() before call')

        clusters = [np.zeros((2, 0))] * self.centers

        for i in range(self.patterns):
            x_c = np.broadcast_to(x[:, i], (self.centers, 2)).T
            loss = np.linalg.norm(x_c - self.kernel, axis=0)
            pos = int(np.argmin(loss))
            clusters[pos] = np.append(clusters[pos], np.reshape(x[:, i], (-1, 1)))

        clusters = [np.reshape(cluster, (2, -1)) for cluster in clusters]

        return clusters


def som(x, epochs, n_centers, lr, decay_rate, verbose=True):
    ep_iterator = tqdm.trange(epochs, disable=verbose)

    n_inputs, n_patterns = x.shape

    R = np.random.permutation(n_patterns)

    W = x[:, R[:n_centers]]

    for k in ep_iterator:
        for i in tqdm.trange(n_patterns, disable=verbose):
            lr = lr * np.exp(-k / decay_rate)
            x_c = np.broadcast_to(x[:, i], (n_centers, 2)).T
            loss = np.linalg.norm(x_c - W, axis=0)
            mx, pos = np.amin(loss), np.argmin(loss)
            dW = lr * (x[:, i] - W[:, pos])
            W[:, pos] = W[:, pos] + dW

    return W


def som_cluters(x, W):

    n_clusters = W.shape[1]

    n_inputs, n_patterns = x.shape

    clusters = [np.zeros((2, 0))] * n_clusters

    for i in range(n_patterns):
        x_c = np.broadcast_to(x[:, i], (n_clusters, 2)).T  # (X(i,:).' * ones(1, n_clusters)).' vedere perchè moltiplica per ones
        loss = np.linalg.norm(x_c - W, axis=0)
        pos = int(np.argmin(loss))  # CHECK AXIS
        clusters[pos] = np.append(clusters[pos], np.reshape(x[:, i], (-1, 1)))

    clusters = [np.reshape(cluster, (2, -1)) for cluster in clusters]

    return clusters

