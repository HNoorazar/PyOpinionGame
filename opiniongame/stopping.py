# functions used for calculating stopping criteria

import numpy as np

######################                     ###################
######################  Stopping Functions ###################
######################                     ###################
##
##  Polarization stop. 
##  This function looks at the population. If sum of number of people 
##  whose opinions are more than .95 and less than .05 is the same as 
##  population, then it would send an stopping signal out!
##
def consensusStopPolarizationStop(currentOpinion, staticConfigurationStuff):
    stopSignal = True
    polar_vector = np.squeeze(currentOpinion)
    polar_vector = list(polar_vector)
    if (sum(i >= .95 for i in polar_vector) + sum(i <= .05 for i in polar_vector)) == staticConfigurationStuff.popSize:
        polarSignal = False
    
    if np.max(currentOpinion) - np.min(currentOpinion) < .03:
        consensusSignal = False
    stopSignal = (polarSignal or consensusSignal)
    return stopSignal
##
##  This functions is stopping tool by looking at total 
##  change that all people have made during the last step of the game.
##
def totalChangeStop(change, staticConfiguration ):
    stopSignal = True
    if change < staticConfiguration.threshold:
        stopSignal = False
    return stopSignal

##
## Window Stop Function here:
##
def windowStop( iterationCount, windowLength, controlLength,
                stateEvol, KevinsMatrix, KevinsCollapsedMatrix  ):
    stopSignal = True
    if iterationCount >= ( windowLength - 1):
        K_count = (iterationCount + 1) % controlLength
        KevinsMatrix [ :, K_count, :] = ( stateEvol[iterationCount + 1 - windowLength : iterationCount + 1, : ,:].max( axis = 0) -
                                          stateEvol[iterationCount + 1 - windowLength : iterationCount + 1, : ,:].min( axis = 0) )
                                           
        whatever_Matrix =  KevinsMatrix.max(axis = 1) - KevinsMatrix.min(axis = 1)
        find_indexes = np.where(whatever_Matrix < 0.01)
        find_ind_array = np.asarray(find_indexes)
        KevinsCollapsedMatrix[find_ind_array[0], find_ind_array[1]] = 0 
        if np.all(KevinsCollapsedMatrix == False):
            stopSignal = False
    return stopSignal   

def iterationStop(config, state, change, iterationNo):
    """
    Iteration count stop function, would terminate the process if
    a certain number of steps is taken.
    """
    if iterationNo >= config.iterationMax:
        return False
    else:
        return True

def individualsChange(config, state, change, iterationNo):
    # TODO: rename Kthreshold to a more meaningful name
    if np.all(change < config.Kthreshold):
        return False
    else:
        return True
    
##
##  The following function terminates the game, 
##  if each the average change in the last time step
##  is less than a threshold.
##  ctState is current state.
##

def averageChange(ctState, constConfiguration):
    stopSignal = True
    K2_all_changes = np.abs(ctState[-1] - ctState[-3]) 
    if (np.sum(K2_all_changes )/constConfiguration.popSize) < (constConfiguration.threshold):
        stopSignal = False
    return stopSignal


##
##  This function compares the changes made in the last time step,
##  with the changes made at step (time - 20). 
##  Generates stopping signal if the difference is less than a threshold.
def HosseinStop(itrNo, state, staticConfigurate ):
    stopSignal = True
    if itrNo > 22:
        H2_all_changes =  state[-2,:,:] - state[-20,:,:]
        if  np.linalg.norm(H2_all_changes < staticConfigurate.Hthreshold):
            stopSignal = False
        return stopSignal


##
##  This function compares two matrices,
##  one matrix is opinion evolution in the last 20 steps,
##  second matrix is opinion evolution since 40 steps ago up to 20 steps ago,
##  and returns norm of difference between those two matrices.
##
def norm_stop(Histr):
    stopSignal = True
    time = np.shape(Histr)[0] - 1
    if time > 40:
        matrix =  Histr[time - 20 : - 1, :, : ] - Histr[time - 40 : time - 20, :, : ]
        if np.linalg.norm( matrix ) < 0.2:
            stopSignal = False
    return stopSignal  

##
##  This function, for each individual and each topic 
##  computes the difference between the maximum and minimum opinion
##  in the last 50 time steps, and if all of those differences is less than
##  threshold 0.0001, sends a termination signal out!
##
def diff_stop(Histry, configuration, diffVec):
    stopSignal = True
    # Histry is not miss-spelled :D just makining sure not to pass around the same variable!
    if np.shape(Histry)[0] >= 51:
        for nt in range(configuration.ntopics):
            for pp in range(configuration.popSize):
                difference = np.max(Histry[-50:-1, pp, nt]) - np.min(Histry[-50:-1, pp, nt])
                if difference < 0.0001:
                    diffVec[pp] = 0
        if np.all(diffVec == False):
                stopSignal = False          
    return stopSignal

