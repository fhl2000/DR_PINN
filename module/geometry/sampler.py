__all__ = ["sample","BatchSampler","random_bbox_sampler","uniform_bbox_sampler"]

import numpy as np
import itertools

def random_bbox_sampler(n,dim,bbox,random="pseudo"):
  assert dim==bbox.shape[1],"bbox isn't in a correct form"
  x_min,x_max=bbox
  x = sample(n,dim,sampler=random)
  x= x_min[None,:]+x*(x_max-x_min)[None,:]
  return x

def uniform_bbox_sampler(n,dim,bbox):
  # not neccesaryly sample exactly n points given the bbox
  assert dim==bbox.shape[1],"bbox isn't in a correct form"
  
  volume=np.prod(bbox[1,:]-bbox[0,:])
  dx = (volume / n) ** (1 / dim)
  pool=[]
  for i in range(dim):
    xi_min,xi_max=bbox[:,i]
    ni=int(np.ceil((xi_max-xi_min)/ dx))
    pool.append(np.linspace(xi_min,xi_max,ni,endpoint=True))
  #produce Cartesian product
  return np.array(list(itertools.product(*pool)))


def sample(n_samples, dimension, sampler="pseudo"):
    """Generate pseudorandom or quasirandom samples in [0, 1]^dimension.

    Args:
        n_samples (int): The number of samples.
        dimension (int): Space dimension.
        sampler (string): One of the following: "pseudo" (pseudorandom), "LHS" (Latin
            hypercube sampling), "Halton" (Halton sequence), "Hammersley" (Hammersley
            sequence), or "Sobol" (Sobol sequence).
    """
    if sampler == "pseudo":
        return pseudorandom(n_samples, dimension)
    if sampler in ["LHS", "Halton", "Hammersley", "Sobol"]:
        return quasirandom(n_samples, dimension, sampler)
    raise ValueError("f{sampler} sampling is not available.")


def pseudorandom(n_samples, dimension):
    """Pseudo random."""
    # If random seed is set, then the rng based code always returns the same random
    # number, which may not be what we expect.
    # rng = np.random.default_rng(config.random_seed)
    # return rng.random(size=(n_samples, dimension), dtype=config.real(np))
    return np.random.random(size=(n_samples, dimension)).astype(np.float32)


def quasirandom(n_samples, dimension, sampler):
    # Certain points should be removed:
    # - Boundary points such as [..., 0, ...]
    # - Special points [0, 0, 0, ...] and [0.5, 0.5, 0.5, ...], which cause error in
    #   Hypersphere.random_points() and Hypersphere.random_boundary_points()
    try:
        import skopt,pyDOE   
    except :
        raise ModuleNotFoundError("package 'skopt' or 'pyDOE'  hasn't been installed, please pip install scikit-optimize pyDOE")
    skip = 0
    if sampler == "LHS":
        if dimension<=2:
            sampler = skopt.sampler.Lhs(lhs_type="centered",criterion="maximin", iterations=1000)
        else:  # skopt.sampler.Lhs() is ill for high dimension sampling, use pyDOE.lhs() instead.
            return pyDOE.lhs(dimension,n_samples)
    elif sampler == "Halton":
        # 1st point: [0, 0, ...]
        sampler = skopt.sampler.Halton(min_skip=1, max_skip=1)
    elif sampler == "Hammersley":
        # 1st point: [0, 0, ...]
        if dimension == 1:
            sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
        else:
            sampler = skopt.sampler.Hammersly()
            skip = 1
    elif sampler == "Sobol":
        # 1st point: [0, 0, ...], 2nd point: [0.5, 0.5, ...]
        sampler = skopt.sampler.Sobol(randomize=False)
        if dimension < 3:
            skip = 1
        else:
            skip = 2
    space = [(0.0, 1.0)] * dimension
    return np.asarray(
        sampler.generate(space, n_samples + skip)[skip:], dtype=np.float32
    )



class BatchSampler:
    """Samples a mini-batch of indices.

    The indices are repeated indefinitely. Has the same effect as:

    .. code-block:: python

        indices = tf.data.Dataset.range(num_samples)
        indices = indices.repeat().shuffle(num_samples).batch(batch_size)
        iterator = iter(indices)
        batch_indices = iterator.get_next()

    However, ``tf.data.Dataset.__iter__()`` is only supported inside of ``tf.function`` or when eager execution is
    enabled. ``tf.data.Dataset.make_one_shot_iterator()`` supports graph mode, but is too slow.

    This class is not implemented as a Python Iterator, so that it can support dynamic batch size.

    Args:
        num_samples (int): The number of samples.
        shuffle (bool): Set to ``True`` to have the indices reshuffled at every epoch.
    """

    def __init__(self, num_samples, shuffle=True):
        self.num_samples = num_samples
        self.shuffle = shuffle

        self._indices = np.arange(self.num_samples)
        self._epochs_completed = 0
        self._index_in_epoch = 0

        # Shuffle for the first epoch
        if shuffle:
            np.random.shuffle(self._indices)

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def get_next(self, batch_size):
        """Returns the indices of the next batch.

        Args:
            batch_size (int): The number of elements to combine in a single batch.
        """
        if batch_size > self.num_samples:
            raise ValueError(
                "batch_size={} is larger than num_samples={}.".format(
                    batch_size, self.num_samples
                )
            )

        start = self._index_in_epoch
        if start + batch_size <= self.num_samples:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._indices[start:end]
        else:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_samples = self.num_samples - start
            indices_rest_part = np.copy(
                self._indices[start : self.num_samples]
            )  # self._indices will be shuffled below.
            # Shuffle the indices
            if self.shuffle:
                np.random.shuffle(self._indices)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_samples
            end = self._index_in_epoch
            indices_new_part = self._indices[start:end]
            return np.hstack((indices_rest_part, indices_new_part))