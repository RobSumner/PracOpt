"""A set of useful functions for use in optimisation problems."""

import numpy as np

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

def evaluate(Algorithm, runs=25):
   """Evaluate the performance of an optimiser class.
      Parameters:
      SimAnneal - Simulated annealing class. 
      runs - Number of separate optimise runs to perform. 
      Returns:
      performance_data - Averaged data on performance over iteration 
                         and time."""

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

   return np.concatenate((iters, t_average, f_average), axis=1)