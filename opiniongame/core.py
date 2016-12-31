import numpy as np

def pick_topic(nTopics):
    return np.random.randint(nTopics)

def interaction_update(oldS, oldH, speaker_potential, hearer_potential, learning_rate):
    """
    Input: opinion state of speaker and hearer, potential functions for each individual, learning rate
    Output: updated opinion state of each, and delta for each.
    """
    diff = oldS - oldH

    speaker_delta = (learning_rate / 2.0) * speaker_potential(abs(diff)) * diff
    hearer_delta = (learning_rate / 2.0) * hearer_potential(abs(diff)) * diff

    x = oldS - speaker_delta

    if x > 1.0:
        (newS, dS) = (1.0, 1.0 - oldS)
    elif x < 0.0:
        (newS, dS) = (0.0, 0.0 - oldS)
    else:
        (newS, dS) = (x, -speaker_delta)

    x = oldH + hearer_delta

    if x > 1.0:
        (newH, dH) = (1.0, 1.0 - oldH)
    elif x < 0.0:
        (newH, dH) = (0.0, 0.0 - oldH)
    else:
        (newH, dH) = (x, hearer_delta)

    return (newS, dS, newH, dH)

def handle_pair( statConfigs, dynState, currentOpinionMatrix, s, h ):
    print("==> HANDLE PAIR")
    oldS = currentOpinionMatrix[s, :]
    oldH = currentOpinionMatrix[h, :]
    
    topic = pick_topic(statConfigs.ntopics)
    
    (newS, dS, newH, dH) = interaction_update(oldS[topic], oldH[topic], 
                                              speaker_func, hearer_func, 
                                              statConfigs.learning_rate)

    # Record all changes happening to each person, each topic, )
    wS = dynState.couplingWeights[s,topic,:]
    wH = dynState.couplingWeights[h,topic,:]
    chgS = 0.0
    chgH = 0.0
    
########    Vectorized of the next part      ##########################   
########    Why vectorization does not work?
#    newval = oldS + dS * wS
#    newval[newval > 1 ] = 1
#    newval[newval < 0 ] = 0
#    chgS = np.sum( np.abs(opinions[s,:] - newval) )

#    newval = oldH + dH * wH
#    newval[newval > 1 ] = 1
#    newval[newval < 0 ] = 0
#    chgH = np.sum( np.abs(opinions[h,:] - newval) )
##################################################################
    for i in range(len(oldS)):
        newval = oldS[i] + dS * wS[i]
        if newval > 1:
            newval = 1
        elif newval < 0:
            newval = 0
        chgS = chgS + np.abs(currentOpinionMatrix[s,i] - newval)
        currentOpinionMatrix[s,i] = newval
        newval = oldH[i] + dH * wH[i]
        if newval > 1:
            newval = 1
        elif newval < 0:
            newval = 0
        currentOpinionMatrix[h,i] = newval
        chgH = chgH + np.abs( currentOpinionMatrix[h,i] - newval )       
##################################################################        

    return (currentOpinionMatrix, chgS + chgH)    

def one_step(config, game_state, ufunc, opinionsO):
    """ one_step entails updating as many pairs of individuals as allowed.
        input opinionO here is 2D
    """
    print("==> ONE STEP")

    pairs = ufunc.selector(game_state.adj)

    print("OPINIONSO: "+str(opinionsO))
    print("PAIRS: "+str(pairs))

    # Record all changes happening to each person, each topic, 
    # total_change is total change happening in one single time step!
    total_change = 0.0
    for i in range(pairs.shape[0]):
        (opinionsO, c) = handle_pair(config, game_state, opinionsO, pairs[i,0], pairs[i,1])
        total_change = total_change + c

    # Output opinions has to be 3D to be used by "concatenate"
    opinions3d = np.zeros((1, np.shape(opinionsO)[0], np.shape(opinionsO)[1] ))
    opinions3d[0,:,:] = opinionsO
    return (opinions3d, total_change, pairs)

def run_until_convergence(config, dynamicConfig, userFunction):
    print("==> RUN_UNTIL_CONVERGENCE")

    # initialize history with 3D array at time =0
    state = np.array([dynamicConfig.opinions], copy=True)
    state = np.concatenate((state, [dynamicConfig.opinions]), axis=0)

    print("STATE= "+str(state))

    # initiate matrix of changes with infinity at t=0 for stopping purposes.
    # This is a matrix whose rows are people and each column corresponds to a topic.
    K2_all_changes = 10 * np.ones((config.popSize, config.ntopics))

    convolutionGoVector = np.ones((config.popSize, 1))
    diffVector = np.ones((config.popSize, 1))

    TerminationSignal = True
    itrCount = 0
    window_length = 20
    control_length = 30
    Kevins_matrix =  100 * np.ones(( config.popSize,  control_length, config.ntopics))
    Kevins_collapsed_matrix = np.ones(( config.popSize, config.ntopics ))
    
    while TerminationSignal:
        # the input in the next line is 2D.
        (outOpinions, chg, all_pairs) = one_step(config, dynamicConfig, userFunction, state[-1])
        state = np.concatenate((state, outOpinions), axis = 0)
        itrCount = itrCount + 1
        
        if SelPoten['stop'] == 'windowStop' :
            TerminationSignal = windowStop( itrCount, window_length, control_length,
                                      state, Kevins_matrix, Kevins_collapsed_matrix )
        elif SelPoten['stop'] == 'iterationStop':
            TerminationSignal = iterationStop (staticConfiguration.iterationMax, itrCount)
            
        elif SelPoten['stop'] == 'totalChangeStop': 
            TerminationSignal = totalChangeStop(chg, staticConfiguration)
            
        elif SelPoten['stop'] == 'averageChange':
            TerminationSignal = averageChange(state, staticConfiguration, K2_all_changes)
            
        elif SelPoten['stop'] == 'norm_stop':
            TerminationSignal = norm_stop(state)
            
        elif SelPoten['stop'] == 'diff_stop' :
            TerminationSignal = diff_stop(state, staticConfiguration, diffVector)
            
        elif SelPoten['stop'] == 'conv_stop' :
            TerminationSignal = conv_stop(itrCount, state, staticConfiguration, convolutionGoVector)
            
        elif SelPoten['stop'] ==  'HosseinStop':
            TerminationSignal = HosseinStop(itrCount, state, staticConfiguration )
                
        elif SelPoten['stop'] == 'consensusStopPolarizationStop' :
            TerminationSignal = consensusStopPolarizationStop( outOpinions, staticConfiguration )
            
    state = np.delete(state,-1, axis = 0)
    return state
