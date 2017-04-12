# First Driver for Community Identity

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import opiniongame.config as og_cfg
import opiniongame.IO as og_io
import opiniongame.coupling as og_coupling
import opiniongame.state as og_state
import opiniongame.opinions as og_opinions
import opiniongame.adjacency as og_adj
import opiniongame.selection as og_select
import opiniongame.potentials as og_pot
import opiniongame.core as og_core
import opiniongame.stopping as og_stop
import numpy as np
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
uppBound_list = np.arange(0.000001, 0.11, 0.005)

# List of uniqueness Strength parameter
individStrength = np.arange(0.0001, 0.00051, 0.0001) 

noInitials = np.arange(1) # Number of different initial opinions.
noGames = np.arange(1)    # Number of different game orders.

config.learning_rate = 0.1
# config.uniqStrength = .0002

tau = 0.62
config.iterationMax = 6000

config.printOut()
#
# functions for use by the simulation engine
#
ufuncs = og_cfg.UserFunctions(og_select.PickTwoWeighted,
                              og_stop.iterationStop,
                              og_pot.createTent(tau))
# Run experiments with different adjacencies, different initials, and different order of games.
for uniqForce in individStrength:
    config.uniqStrength = uniqForce
    for upperBound in uppBound_list:
        # Generate different adjacency matrix with different prob. of interaction
        # between different communities
        state.adj = og_adj.CommunitiesMatrix(communityPopSize, numberOfCommunities, upperBound)
    
        if upperBound > 0.0001: 
            config.iterationMax = 6000
        if upperBound > 0.001: 
            config.iterationMax = 6000  

        for countInitials in noInitials:
            # for each adjacency, generate 100 different initial opinions
    #        state.initialOpinions = og_opinions.initialize_opinions(config.popSize, config.ntopics)
         
            # Pick three communities with similar opinions to begin with!
            state.initialOpinions = np.zeros((config.popSize, 1))
            state.initialOpinions[0:25]  = np.random.uniform(low=0.0, high=.25, size=(25,1))
            state.initialOpinions[25:50] = np.random.uniform(low=0.41, high=.58, size=(25,1))
            state.initialOpinions[50:75] = np.random.uniform(low=0.74, high= 1, size=(25,1))
   
            state.couplingWeights = og_coupling.weights_no_coupling(config.popSize, config.ntopics)
            all_experiments_history = {}

            for gameOrders in noGames:
                state = og_core.run_until_convergence(config, state, ufuncs)
                print "One Experiment Done", "gameOrders = " , gameOrders+1, "countInitials=", countInitials+1
                all_experiments_history[ 'experiment' + str(gameOrders+1 )] = state.history
            og_io.saveMatrix('uB ' + str(upperBound) + ' * initCount ' + str(countInitials+1) + 
                             ' * uniqStrength ' + str(config.uniqStrength) +'.mat', all_experiments_history)

# ToDo: short version: We should compare parameters provided by Driver.py and default.
#       just like from command line vs. default.

# TODO: adjacency is provided here, and popSize is changed manually,
# but still, in the code and selection, we have problem.
# population size is not changed there
# in the code, popSize is assumed 20. (initialOpinions are 20-by-20).
# we have to fix that! initialOpinion should have a proper size, based on
# adjacency provided in the driver.


# ToDo : Stuff being printer on the command lines are nor correct
# if we provide stuff in the driver.