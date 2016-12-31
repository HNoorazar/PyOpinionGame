import numpy as np

def initialize_opinions(pop_size, ntopics):
    """
    Given a population size and topic count, generate a set of random
    opinions.  Assumes each topic space is the scalars over [0,1].
    """
    return np.random.rand(pop_size, ntopics)
    