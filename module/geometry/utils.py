import numpy as np
from functools import wraps


def vectorize(**kwargs):
    """numpy.vectorize wrapper that works with instance methods.

    References:

    - https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
    - https://stackoverflow.com/questions/48981501/is-it-possible-to-numpy-vectorize-an-instance-method
    - https://github.com/numpy/numpy/issues/9477
    """

    def decorator(fn):
        vectorized = np.vectorize(fn, **kwargs)

        @wraps(fn)
        def wrapper(*args):
            return vectorized(*args)

        return wrapper

    return decorator