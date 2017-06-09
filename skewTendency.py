# First Driver for Community Identity

import numpy as np

#import cProfile
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

config.threshold = 0.01
config.printOut()
#
# seed PRNG: must do this before any random numbers are
# ever sampled during default generation
#
print(("SEEDING PRNG: "+str(config.startingseed)))
np.random.seed(config.startingseed)
state = og_state.WorldState.fromCmdlineArguments(cmdline, config)
#
# run
#
numberOfCommunities = 3
communityPopSize    = 25
config.popSize = numberOfCommunities * communityPopSize

# List of upper bound probability of interaction between communities
uppBound_list = np.array([.001, 0.002, 0.003, 0.004, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.016, 0.019])
#
# List of uniqueness Strength parameter
#
individStrength = np.arange(0.00001, 0.000451, 0.00006)
individStrength = np.append(0, individStrength)
individStrength = np.array([0.0])
config.skewstrength = 300.0

config.learning_rate = 0.1
tau = 0.62
config.iterationMax = 30000
config.printOut()
#
# functions for use by the simulation engine
#
ufuncs = og_cfg.UserFunctions(og_select.PickTwoWeighted,
                              og_stop.iterationStop,
                              og_pot.createTent(tau))
                              
noInitials = np.arange(1) # Number of different initial opinions.
noGames = np.arange(75)    # Number of different game orders.
# Run experiments with different adjacencies, different initials, and different order of games.
for uniqForce in individStrength:
    config.uniqstrength = uniqForce
    for upperBound in uppBound_list:
        # Generate different adjacency matrix with different prob. of interaction
        # between different communities
        state.adj = og_adj.CommunitiesMatrix(communityPopSize, numberOfCommunities, upperBound)
        print"(upperBound, uniqForce) = (", upperBound, "," , uniqForce , ")"            
        for countInitials in noInitials:
            # for each adjacency, generate 100 different initial opinions
            # state.initialOpinions = og_opinions.initialize_opinions(config.popSize, config.ntopics)
         
            # Pick three communities with similar opinions to begin with!
            state.initialOpinions = np.zeros((config.popSize, 1))
            state.initialOpinions[0:25]  = np.random.uniform(low=0.08, high=.1, size=(25,1))
            state.initialOpinions[25:50] = np.random.uniform(low=0.49, high=.51, size=(25,1))
            state.initialOpinions[50:75] = np.random.uniform(low=0.9, high= .92, size=(25,1))
   
            state.couplingWeights = og_coupling.weights_no_coupling(config.popSize, config.ntopics)
            all_experiments_history = {}

            print "countInitials=", countInitials + 1
            print "config.skewstrength = 300.0", config.skewstrength
            
            for gameOrders in noGames:
                #cProfile.run('og_core.run_until_convergence(config, state, ufuncs)')
                state = og_core.run_until_convergence(config, state, ufuncs)
                state.history = state.history[0:state.nextHistoryIndex,:,:]
                idx_IN_columns = [i for i in xrange(np.shape(state.history)[0]) if (i % (config.popSize)) == 0]
                state.history = state.history[idx_IN_columns,:,:]
                all_experiments_history[ 'experiment' + str(gameOrders+1)] = state.history
            og_io.saveMatrix('uB' + str(upperBound) + '*uS' + str(config.uniqstrength) + 
                             '*initCount' + str(countInitials+1) + '.mat', all_experiments_history)
