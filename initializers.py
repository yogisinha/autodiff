import numpy as np

class Xavier_Initializer():
    def __call__(self, shape):
        return np.random.randn(shape[0], shape[1]) * np.sqrt(2/shape[1])

class Zeros_Initializer():
    def __call__(self, shape):
        return np.zeros(shape)

class Random_Initializer():
    def __call__(self, shape):
        return np.random.randn(*shape)


def random_initializer():
    return random_initializer_fn


def zeros_initializer():
    return zeros_initializer_fn


def xavier_initializer():
    return xavier_initializer_fn

xavier_initializer_fn = Xavier_Initializer()
zeros_initializer_fn = Zeros_Initializer()
random_initializer_fn = Random_Initializer()
