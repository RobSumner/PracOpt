"""Implement optimisation methods

Classes
-------
TrialMode - Enumerated class to describe trial update mode. 
InitialTempMode - Enumerated class to describe initial temperature mode.
SimAnneal - Class to perform Simulated Annealing optimisation

"""

from optimiser import objective
from optimiser.archive import Archive
from enum import Enum
import numpy as np 
import copy

class TrialMode(Enum):
   """Mode describing which trial update mode to use."""
   BASIC = 1
   CHOLESKY = 2
   PARKS = 3

class InitialTempMode(Enum):
   """Mode describing the initial temperature calculation."""
   PRESET = 1
   KIRKPATRICK = 2
   WHITE = 3

class SimAnneal:
   """Perform Simulated Annealing Optimisiation
   Parameters
   ----------
   objective   - Objective class describing function to be optimised.
   trial_mode - The method used for selecting new trial solutions. 
   initial_temp_mode - The method used to select the initial temperature. 

   Public Methods
   -------------- 
   run - Perform the optimisation.
   reset - Reset all variable parameters, storage and classes.  
   acceptable_solution - Implements solution acceptance criteria for 
                           simulated annealing.
   new_trial_solution  - Return a new trial solution based upon trial mode. 
   update_temperature  - Update the annealing temperature. 
   set_initial_temp    - Set the initial temperature based upon initial 
                           temperature mode. 
   uniform_random      - Return 1D sample from uniform variable in range
                           x_min, x_max.  
   """

   def __init__(self, objective, trial_mode, initial_temp_mode):
      """Initialise the optimiser.
         Parameters:
         trial_mode - Defines how the trial solutions are generated.
         initial_temp_mode - Defines how initial temperature is set."""
      # Check input arguments
      if type(trial_mode) != TrialMode:
         raise ValueError('Trial Mode not recognised.')
      self.trial_mode = trial_mode
      if type(initial_temp_mode) != InitialTempMode:
         raise ValueError('Initial Temperature Mode not recognised.')
      self.initial_temp_mode = initial_temp_mode

      # Set objective function
      self.objective = objective
      self.dimension = objective.n

      # Point, Objective value & change in objective.
      self.x = np.zeros((self.dimension, 1)) # Sample point

      # Annealing schedule
      self.initial_T = 10e10
      self.current_T = 10e10
      self.trials = 0 # Total length of chain
      self.acceptances = 0 # Total number of acceptances
      self.decrement_length = 100
      self.max_evaluations = 10000 

      # Archive
      self.archive = Archive()

   def run(self):
      """Perform the optimisation."""
      # Set the initial temperature for process.
      self.set_initial_temp()

      # Set random start point
      self.x = self.uniform_random(dim=self.dimension)

      # Search up to 10000 objective function evaluations
      while self.objective.evaluations < self.max_evaluations:
         # Display progress
         progress_bar(self.objective.evaluations, self.max_evaluations)

         # Find a new acceptable solution
         f0 = self.objective.f(self.x)
         x1 = self.new_trial_solution()
         f1 = self.objective.f(x1)
         while not self.acceptable_solution(f1 - f0):
            x1 = self.new_trial_solution()
            f1 = self.objective.f(x1)
         self.x = x1 

         # Add new solution to archive 
         self.archive.add(x1, f1, self.objective.evaluations)

         # Update temperature following annealing schedule
         self.update_temperature()
      
      # Reset output from carriage return
      print("") 

   def reset(self):
      """Reset variable parameters and sub-classes."""
      # Point, Objective value & change in objective.
      self.x = np.zeros((self.dimension, 1)) # Sample point

      # Annealing schedule
      self.initial_T = 10e10
      self.current_T = 10e10
      self.trials = 0 # Total length of chain
      self.acceptances = 0 # Total number of acceptances

      # Reset archive & objective
      self.archive.reset()
      self.objective.reset()
      
   def acceptable_solution(self, df):
      """Return True if solution decreases objective function.
         If objective function increases, return true following the 
         simulated annealing acceptance probability. 
         Random number generation is used to implement random acceptance.
         Parameters:
         df - The change in objective function value for solution.
         Returns:
         Bool - Describing if solution should be accepted."""
      if df < 0:
         # Always accept if f decreases.
         self.acceptances += 1
         return True
      else:
         p_accept = np.exp(-1*df/self.current_T)
         sample = self.uniform_random(0,1)
         if sample <= p_accept:
            self.acceptances += 1
            return True
         else:
            return False

   def new_trial_solution(self, x0=None):
      """Return new trial solution following trial mode.
         Parameters:
         x0 - Starting point for the new trial x. If x0 is not provided,
               the current self.x value is used.
         Returns:
         x_new - New trial position of same dimension as input x0."""      
      # Set start point if none provided.
      if x0 is None:
         x0 = self.x
      elif len(x0) != self.dimension:
         raise ValueError('Incorrect starting point dimension for new trial.')

      # Simple diagonal (C) matrix update.
      x_new = np.zeros((self.dimension,1))
      if self.trial_mode is TrialMode.BASIC:
         for i in range(self.dimension):
            # Sample new feasible position from altered range
            # This avoids wasted samples and result is the same.
            x_min = max(self.objective.x_min, x0[i] + self.objective.x_min)
            x_max = min(self.objective.x_max, x0[i] + self.objective.x_max)
            x_new[i] = self.uniform_random(x_min=x_min, x_max=x_max)
            while not self.objective.is_feasible(x_new, i):
               x_new[i] = self.uniform_random(x_min=x_min, x_max=x_max)

         self.trials += 1
      return x_new

   def update_temperature(self):
      """Update the annealing temperature."""
      # Implement basic length based decrement. 
      self.current_T = self.initial_T*0.95**(self.trials // self.decrement_length)
      return

   def set_initial_temp(self):
      """Set the initial temperature based upon initial_temp_mode:
         KIRKPATRIC - Based on Kirkpatrick [1984] 
                    - Set T0 so av. prob. of solution increasing f
                      is ~0.8.
         WHITE - Based on White [1984] 
               - Set T0 = sigma where sigma issd of variation in 
                 objective function during initial search.
         PRESET - Use a constant value preset in the class.
                - Default value is 10e10 in this case."""
      if self.initial_temp_mode is InitialTempMode.PRESET:
         # Temp already set in class - no calculation needed.
         return
      
      # Calculate T0 following White/Kirkpatrick
      samples = 100
      df_samples = np.zeros((samples,1))
      for i in range(samples):
         # Random starting point in range [-2,2]
         x0 = self.uniform_random(dim=self.dimension)
         f0 = self.objective.f(x0)
         
         # Accept any move which increase value of f.  
         x1 = self.new_trial_solution(x0)
         f1 = self.objective.f(x1)
         # Limit to 10 trials for speed.
         for _ in range(10):
            if f1 <= f0:
               x1 = self.new_trial_solution(x0)
               f1 = self.objective.f(x1)
            else:
               df_samples[i] = f1 - f0
      # Extract zeros from array (failed trials)
      df_samples = df_samples[df_samples != 0]

      # Set T0 based on mode. 
      if self.initial_temp_mode is InitialTempMode.KIRKPATRICK:
         # Set T to be average
         self.initial_T = -1*np.mean(df_samples)/np.log(0.8)
      elif self.initial_temp_mode is InitialTempMode.WHITE:
         # Calculate sd of samples
         self.initial_T = np.std(df_samples) 
      
      # Store current temp as initial temp
      self.current_T = self.initial_T

      # Resent number of evaluations to 0 
      self.objective.reset()
      
   def uniform_random(self, x_min=None, x_max=None, dim=1):
      """Return nD sample from scaled uniform random variable.
         Parameters:
         [x_min, x_max] - The range for generated numbers. This defaults
                           to the provided objective function values. 
         dim - Dimension of random sample to be generated.
         Returns:
         x - Randomly generated number or array of numbers of dimension dim.
         """
      # Set default values
      if x_min is None:
         x_min = self.objective.x_min
      if x_max is None:
         x_max = self.objective.x_max

      # Check the range of values. 
      if x_min >= x_max:
         raise ValueError("Incorrect random number range.")

      # Generate number
      x_range = x_max - x_min
      return (np.random.rand(dim,1)*x_range) + x_min   

def progress_bar(value, max_value, width=15):
   """Print a progress bar utilising the carriage return function.
      value - Number representing the current progress of process.
      Parameters:
      max_value - Maximum possible value in process 
      width - Number of characters in the progress bar."""
   progress = round(value/max_value*width)
   remaining = width - progress
   print('\rOptimisation Progress: ' + "+"*progress + "-"*remaining, end="")

def evaluate(SimAnneal, runs=25):
   """Evaluate the performance of an optimiser class.
      Parameters:
      SimAnneal - Simulated annealing class. 
      runs - Number of separate optimise runs to perform. 
      Returns:
      performance_data - Averaged data on performance over iteration 
                         and time."""

   max_evals = SimAnneal.max_evaluations
   f_data = np.zeros((max_evals, runs))
   time_data = np.zeros((max_evals, runs))

   for i in range(runs):
      # Reset all values
      SimAnneal.reset()

      # Run the optimisation
      print("Analysis run: ", i)
      SimAnneal.run()

      # Get objective data
      data = SimAnneal.archive.objective_data(max_evals)
      f_data[:,i] = data[:,2]
      time_data[:,i] = data[:,1]
   
   # Average across runs.
   f_average = np.reshape(np.mean(f_data, axis=1), (max_evals,1))
   t_average = np.reshape(np.mean(time_data, axis=1), (max_evals,1))
   iters = np.reshape(np.linspace(1, max_evals, max_evals), (max_evals,1))

   return np.concatenate((iters, t_average, f_average), axis=1)
   