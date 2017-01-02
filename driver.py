# test driver to verify that new version of code works
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
# TODO: add option to generate defaults and save to file
# TODO: interpret args to get filename if specified on cmd line
config = og_cfg.staticParameters()
config.readFromFile('staticParameters.cfg')

config.printOut()

#
# seed PRNG: must do this before any random numbers are
# ever sampled during default generation
#
print("SEEDING PRNG: "+str(config.startingseed))
np.random.seed(config.startingseed)

state = og_state.WorldState.fromCmdlineArguments(cmdline, config)

#
# functions for use by the simulation engine
#
ufuncs = og_cfg.UserFunctions(og_select.FastPairSelection,
                              og_stop.iterationStop,
                              og_pot.createTent(0.5))

#
# run
#
polarized = 0
notPolarized = 0

for i in range(100):
    state = og_core.run_until_convergence(config, state, ufuncs)
    results = og_opinions.isPolarized(state.history[-1], 0.05)
    for result in results:
        if result:
            polarized += 1
        else:
            notPolarized += 1
    state.reset()
    state.initialOpinions = og_opinions.initialize_opinions(config.popSize, config.ntopics)

print((polarized, notPolarized))