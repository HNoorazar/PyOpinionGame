""" Functions used for calculating stopping criteria.

A stopping function will return True if it is time to stop,
False otherwise.
"""
 # TODO: In the long paper, we used "windows stop" 
 # which was the most effective way. but it is not here.
 # I even tried the convolution thing, but even that did not work.
import numpy as np

def totalChangeStop(config, state, change, iterationNo):
    """
    Stop if the total change has dropped below some threshold.
    """
    if np.sum(np.abs(change)) < config.threshold:
        return True
    else:
        return False

def iterationStop(config, state, change, iterationNo):
    """
    Iteration count stop function, would terminate the process if
    a certain number of steps is taken.
    """
    if iterationNo >= config.iterationmax:
        return True
    else:
        return False

def allChangeStop(config, state, change, iterationNo):
    """
    Stop if all individuals have changed less than some given
    threshold value.
    """

    # TODO: rename Kthreshold to a more meaningful name
    if np.all(np.abs(change) < config.Kthreshold):
        return True
    else:
        return False

def averageChangeStop(config, state, change, iterationNo):
    """
    Stop if the average change has dropped below a given threshold.
    """
    if (np.sum(np.abs(change))/config.popSize) < config.threshold:
        return True
    else:
        return False

def polarizationStop(config, state, change, iterationNo):
    """
    Stop if 95% of population is polarized.
    """
    v = np.reshape(state.history[state.nextHistoryIndex-1], np.size(state.history[state.nextHistoryIndex-1], 0))
    if (np.sum( i>0.95 for i in v) + np.sum( k<0.05 for k in v ) ) > (.95 * config.popSize):
        return True
    else:
        return False


