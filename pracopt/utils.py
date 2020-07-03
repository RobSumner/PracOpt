"""A set of useful functions for use in optimisation problems."""

import numpy as np
import scipy.io

def progress_bar(value, max_value, width=15):
    """Print progress bar.

    Print a progress bar (utilising the carriage return function).

    Parameters
    ----------
    value : :obj:`float` or :obj:`int`
          Number representing the current progress of process.

    max_value : :obj:`float` or :obj:`int`
        Maximum possible value in process.

    width : :obj:`int`, optional
        Number of characters in the progress bar. Default is 15.

    """
    progress = round(value/max_value*width)
    remaining = width - progress
    print('\rOptimisation Progress: ' + "+"*progress + "-"*remaining, end="")

def evaluate(algorithm, runs=5, filepath=None, description=None):
    """Evaluate optimiser performance.

    Evaluate the performance of an optimiser class. Because of the random
    search nature of optimiser classes, sevveral evaulation runs are
    performed and results are averaged.

    Parameters
    ----------
    algorithms : :class:`pracopt.optimiser.Optimiser`
        The optimiser algorithm class to run.

    runs : :obj:`int`, optional.
        The number of runs to use when evaluating optimiser.

    filepath : :obj:`str`, Optional
        File path to save results to (as .mat file). If None, file is not
        saved.

    description : :obj:`str`, optional.
        Description string to save with results.

    Returns
    -------
    results : :class:`numpy.array`
        Result array. Each row consists of [optimiser step, average time,
        average value].
        I.e. the results are averaged across "runs" by objective function
        step.

    """
    max_evals = algorithm._max_evaluations
    f_data = np.zeros((max_evals, runs))
    time_data = np.zeros((max_evals, runs))

    for i in range(runs):
        algorithm.reset()

        print("Analysis run: ", i)
        algorithm.run()

        # Get objective data
        data = algorithm.archive.objective_data(max_evals)
        f_data[:,i] = data[:,2]
        time_data[:,i] = data[:,1]

    f_average = np.reshape(np.mean(f_data, axis=1), (max_evals,1))
    t_average = np.reshape(np.mean(time_data, axis=1), (max_evals,1))
    iters = np.reshape(np.linspace(1, max_evals, max_evals), (max_evals,1))

    if filepath is not None:
        data_dict = {'f_average':f_average, 't_average':t_average, 'iters':iters}
        if description is not None:
            data_dict['description'] = description
        scipy.io.savemat(filepath + '.mat', data_dict)

    return np.concatenate((iters, t_average, f_average), axis=1)