"""A set of useful functions for use in optimisation problems."""

import numpy as np
import scipy.io

def progress_bar(value, max_value, width=15):
   """Print a progress bar utilising the carriage return function.
      value - Number representing the current progress of process.
      Parameters:
      max_value - Maximum possible value in process 
      width - Number of characters in the progress bar.
   """
   progress = round(value/max_value*width)
   remaining = width - progress
   print('\rOptimisation Progress: ' + "+"*progress + "-"*remaining, end="")

def evaluate(Algorithm, runs=25, filepath=None, description=None):
   """Evaluate the performance of an optimiser class.
      Parameters:
      Algorithm - Algorithm to run. 
      runs - Number of separate optimise runs to perform. 
      filepath - File path to save results to (as .mat file).
      description - String to save with results to describe results. 
      Returns:
      performance_data - Averaged data on performance over iteration 
                         and time.
   """

   max_evals = Algorithm.max_evaluations
   f_data = np.zeros((max_evals, runs))
   time_data = np.zeros((max_evals, runs))

   for i in range(runs):
      # Reset all values
      Algorithm.reset()

      # Run the optimisation
      print("Analysis run: ", i)
      Algorithm.run()

      # Get objective data
      data = Algorithm.archive.objective_data(max_evals)
      f_data[:,i] = data[:,2]
      time_data[:,i] = data[:,1]
   
   # Average across runs.
   f_average = np.reshape(np.mean(f_data, axis=1), (max_evals,1))
   t_average = np.reshape(np.mean(time_data, axis=1), (max_evals,1))
   iters = np.reshape(np.linspace(1, max_evals, max_evals), (max_evals,1))

   # Save file if filename is provided
   if filepath is not None:
      data_dict = {'f_average':f_average, 't_average':t_average, 'iters':iters}
      if description is not None:
         data_dict['description'] = description
      scipy.io.savemat(filepath + '.mat', data_dict)
      
   return np.concatenate((iters, t_average, f_average), axis=1)