import numpy as np

#
# Initial Weight matrix for no coupling
#
def weights_no_coupling( pop_size, ntopics ):
    return np.tile(np.eye( ntopics, ntopics ), ( pop_size, 1, 1 ))
#
#   All people have the same weights between topics here
#   Pay Attention first dimension is population, second is topics
def weights_coupling_same( pop_size, ntopics ):
    weights = (2 * np.random.rand(ntopics,ntopics)) - 1
    signw = np.sign(weights)
    weights = np.multiply(weights, np.transpose(signw))
    np.fill_diagonal(weights,1)
    weights = np.tile(weights, (pop_size, 1, 1))
    return weights
##
##  Initial Weight matrix for coupling, people have different weights
##  between topics here
##  Pay Attention first dimension is population, second is topics
def weights_coupling( pop_size, ntopics ):
    weights = (2 * np.random.rand(pop_size, ntopics,ntopics)) - 1
    for ii in range(pop_size):
        signw = np.sign(weights[ii, :, :])
        weights[ii,:,:] = np.multiply(weights[ii,:,:], np.transpose(signw))
        np.fill_diagonal(weights[ii,:,:],1)
    return weights