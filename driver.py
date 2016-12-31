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

#
# check optional arguments and generate defaults if missing
#
weights = None
initialOpinions = None
adj = None

if cmdline.args.weights is not None:
    weights = og_io.loadNamedMatrix(cmdline.args.weights, 'weights')
else:
    weights = og_coupling.weights_no_coupling(config.popSize, config.ntopics)

if cmdline.args.initialOpinions is not None:
    initialOpinions = og_io.loadNamedMatrix(cmdline.args.initialOpinions, 'initialOpinions')
else:
    initialOpinions = og_opinions.initialize_opinions(config.popSize, config.ntopics)

if cmdline.args.adjacency is not None:
    adj = og_io.loadNamedMatrix(cmdline.args.adjacency, 'adjacency')
else:
    adj = og_adj.make_adj(config.popSize, 'full')

state = og_state.WorldState(adj, weights, initialOpinions)
state.validate()

wPopsize = np.shape(weights)[0]
wNtopics = np.shape(weights)[1]

if wPopsize != config.popSize:
    print("WARNING: popsize from data files disagrees with cfg.")
    config.popSize = wPopsize

if wNtopics != config.ntopics:
    print("WARNING: ntopics from data files disagrees with cfg.")
    config.ntopics = wNtopics

#
# functions for use by the simulation engine
#
ufuncs = og_cfg.UserFunctions(og_select.FastPairSelection,
                              og_stop.iterationStop,
                              og_pot.createTent(0.5, 2.0, -2.0))

state = og_core.run_until_convergence(config, state, ufuncs)

print("FINAL STATE=>"+str(state.history))