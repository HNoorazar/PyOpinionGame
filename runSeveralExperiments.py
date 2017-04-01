import numpy as np
# I think this deserves to go to "core" with default no_of_experiments = 1.
# As user needs to just change one thing, she can just change no_of_experiments in her driver.py.

def run_all_experiments(config, state, ufunc):
    """ This function runs a set of experiments with the same initial opinions.
        Number of Experiments are given in config file.
        Goal is to have different orders of interactions with the same initials.
    """

    # initialize history with a dictionary.
    all_experiments_history = {}
    
    for experiment_no in range(config.no_of_experiments):
        # copy the instance here, for use of refer by refrence thing! 
        # O.W. it would be gone.
        stateCopy = copy.deepcopy(state)
        
        # 1- run out of all experiments:
        history = run_until_convergence(config, state, ufunc)
        
        all_experiments_history[ 'experiment' + str(experiment_no + 1 )] = history
    return all_experiments_history
    
