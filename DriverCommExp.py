# First Driver for Community Identity

import numpy as np

import opiniongame.config as og_cfg
import opiniongame.IO as og_io
import opiniongame.coupling as og_coupling
import opiniongame.state as og_state
import opiniongame.adjacency as og_adj
import opiniongame.selection as og_select
import opiniongame.potentials as og_pot
import opiniongame.core as og_core
import opiniongame.stopping as og_stop
#
# process command line
#
cmdline = og_cfg.CmdLineArguments()
cmdline.printOut()
#
# load configuration
#
config = og_cfg.staticParameters()
config.readFromFile('staticParameters.cfg')

""" What is wrong here???? 
Terminal works here, if I do this two lines in python command window, it works.
But when I run this code via python it cannot read uniqStrength!
What the hell???
"""
print "config.learning_rate", config.learning_rate
config.uniqStrength = 10
print "config.uniqStrength= ", config.uniqStrength
config.threshold = 0.01
config.printOut()
#
# seed PRNG: must do this before any random numbers are
# ever sampled during default generation
#
print("SEEDING PRNG: "+str(config.startingseed))
np.random.seed(config.startingseed)
state = og_state.WorldState.fromCmdlineArguments(cmdline, config)

#
# run
#
numberOfCommunities = 3
communityPopSize    = 25
config.popSize = numberOfCommunities * communityPopSize

# List of upper bound probability of interaction between communities
uppBound_list = np.arange(0.001, 0.0161, 0.003)

# List of uniqueness Strength parameter
individStrength = np.arange(0, 0.1, 0.1)

config.learning_rate = 0.1
tau = 0.62
config.iterationMax = 10000
config.printOut()
#
# functions for use by the simulation engine
#
ufuncs = og_cfg.UserFunctions(og_select.PickTwoWeighted,
                              og_stop.iterationStop,
                              og_pot.createTent(tau))
                              
noInitials = np.arange(1) # Number of different initial opinions.
noGames = np.arange(1)    # Number of different game orders.
# Run experiments with different adjacencies, different initials, and different order of games.
for uniqForce in individStrength:
    config.uniqStrength = uniqForce
    for upperBound in uppBound_list:
        # Generate different adjacency matrix with different prob. of interaction
        # between different communities
        state.adj = og_adj.CommunitiesMatrix(communityPopSize, numberOfCommunities, upperBound)
    
        if upperBound >= 0.01:
            config.iterationMax = 12000
 
        if upperBound <= 0.004:
            config.iterationMax = 8000
            
        for countInitials in noInitials:
            # for each adjacency, generate 100 different initial opinions
            # state.initialOpinions = og_opinions.initialize_opinions(config.popSize, config.ntopics)
         
            # Pick three communities with similar opinions to begin with!
            state.initialOpinions = np.zeros((config.popSize, 1))
            state.initialOpinions[0:25]  = np.random.uniform(low=0.0, high=.25, size=(25,1))
            state.initialOpinions[25:50] = np.random.uniform(low=0.41, high=.58, size=(25,1))
            state.initialOpinions[50:75] = np.random.uniform(low=0.74, high= 1, size=(25,1))
   
            state.couplingWeights = og_coupling.weights_no_coupling(config.popSize, config.ntopics)
            all_experiments_history = {}
            print "(uniqForce, upperBound)  =(", uniqForce, "," , upperBound , ")"
            print "countInitials=", countInitials + 1
            
            for gameOrders in noGames:
                state = og_core.run_until_convergence(config, state, ufuncs)
                print "One Experiment Done" , "gameOrders = " , gameOrders+1
                all_experiments_history[ 'experiment' + str(gameOrders+1)] = state.history
            og_io.saveMatrix('uB' + str(upperBound) + '*uS' + str(config.uniqStrength) + 
                             '*initCount' + str(countInitials+21) + '.mat', all_experiments_history)