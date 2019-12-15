"""Perform Optimisation

Classes
-------
TrialMode - Enumerated class to describe trial update mode. 
InitialTempMode - Enumerated class to describe initial temperature mode.
SimAnneal - Class to perform Simulated Annealing optimisation

"""

from optimiser import objective
from enum import Enum
import numpy as np

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

   Public Methods
   -------------- 
   new_trial_solution - Return a new trial solution based upon trial mode. 
   uniform_random - Return 1D sample from uniform variable in range
                    x_min, x_max. 
   initial_temp - Set the initial temperature. 
   """

   def __init__(self, objective, trial_mode, initial_temp_mode):
      """Initialise the optimiser."""
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
      self.current_T = 10000
      self.max_evaluations = 10000 

   def run(self):
      """Perform the optimisation."""
      # Set the initial temperature for process.
      self.set_initial_temp()

      # Search up to 10000 objective function evaluations
      while self.objective.evaulations < self.max_evaluations:
         # Find a new acceptable solution
         f0 = self.objective.f(self.x)
         x1 = self.new_trial_solution()
         while not self.acceptable_solution(self.objective.f(x1) - f0):
            x1 = self.new_trial_solution()
         self.x = x1 

         # Add new solution to archive 

         # Update temperature following annealing schedule
         self.update_temperature()

      return self.x     
      
   def acceptable_solution(self, df):
      """Return True if solution decreases objective function, or based
         on acceptance probability if f increases. Else, return False."""
      if df < 0:
         # Always accept if f decreases.
         return True
      else:
         p_accept = np.exp(-1*df/self.current_T)
         sample = self.uniform_random(0,1)
         if sample <= p_accept:
            return True
         else:
            return False

   def new_trial_solution(self, x0=None):
      """Return new trial solution following trial mode.
         If given, x0 is used as starting point for the new trial x.
         Otherwise, current self.x value is used."""      
      # Set start point if none provided.
      if x0 is None:
         x0 = self.x
      elif len(x0) != self.dimension:
         raise ValueError('Incorrect starting point dimension for new trial.')

      # Simple diagonal (C) matrix update.
      if self.trial_mode is TrialMode.BASIC:
         x_new = x0
         for i in range(self.dimension):
            # Sample new position until it is feasible
            u_i = self.uniform_random(self.objective.x_min, self.objective.x_max)
            x_new[i] = x0[i] + u_i
            while not self.objective.is_feasible(x_new, i):
               u_i = self.uniform_random(self.objective.x_min, self.objective.x_max)
               x_new[i] = x0[i] + u_i
         return x_new
      else:
         return np.zeros((self.dimension, 1))

   def update_temperature(self):
      """Update the annealing temperature."""
      return

   def set_initial_temp(self):
      """Set the initial temperature.
         Kirkpatrick [1984] - Set T0 so av. prob. of solution increasing f
         is ~0.8.
         White [1984] - Set T0 = sigma where sigma issd of variation in 
         objective function during initial search."""
      if self.initial_temp_mode is InitialTempMode.PRESET:
         # Temp already set in class - no calculation needed.
         return
      
      # Calculate T0 following White/Kirkpatrick
      samples = 100
      x_range = self.objective.x_max - self.objective.x_min
      df_samples = np.zeros((samples,1))
      for i in range(samples):
         # Random starting point in range [-2,2]
         x0 = np.random.rand(self.dimension, 1)*x_range + self.objective.x_min
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
         self.current_T = -1*np.mean(df_samples)/np.log(0.8)
      elif self.initial_temp_mode is InitialTempMode.WHITE:
         # Calculate sd of samples
         self.current_T = np.std(df_samples) 
      
   def uniform_random(self, x_min, x_max):
      """Return 1D sample from scaled uniform random variable."""
      if x_min >= x_max:
         raise ValueError("Incorrect random number range.")

      x_range = x_max - x_min
      return (np.random.rand()*x_range) + x_min   