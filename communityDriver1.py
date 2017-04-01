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

numberOfCommunities = 5
communityPopSize    = 50
uppBound = .3 # upper bound of probability interaction between communities.

adjMat = og_adj.CommunitiesMatrix( communityPopSize , numberOfCommunities , uppBound)

config.learning_rate = 0.1
tau = 0.62

