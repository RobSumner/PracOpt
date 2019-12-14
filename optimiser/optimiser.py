"""Perform Optimisation

Classes
-------
TrialMode - Enumerated class to describe trial update mode. 
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

class SimAnneal:
   """Perform Simulated Annealing Optimisiation
   Parameters
   ----------
   objective   - Objective class describing function to be optimised.

   Public Methods
   -------------- 
   """

   def __init__(self, objective, trial_mode):
      """Initialise the optimiser."""
      self.objective = objective
      self.dimension = objective.n
      self.iteration = 0

      # Point, Objective value & change in objective.
      self.x = np.zeros((self.dimension, 1))
      self.current_f = self.objective.f(self.x)
      self.df = 0
      
      if type(trial_mode) != TrialMode:
         raise ValueError('Trial Mode not recognised.')
      self.trial_mode = trial_mode

   def new_trial(self):
      """Return new trial solution following trial mode."""
      # Simple diagonal (C) matrix update.
      if self.trial_mode is TrialMode.BASIC:
         x_new = self.x
         for i in range(self.dimension):
            # Sample new position until it is feasible
            u_i = self.uniform_random(self.objective.x_min, self.objective.x_max)
            x_new[i] = self.x[i] + u_i
            while not self.objective.is_feasible(x_new, i):
               u_i = self.uniform_random(self.objective.x_min, self.objective.x_max)
               x_new[i] = self.x[i] + u_i
         return x_new
      else:
         return np.zeros((self.dimension, 1))
      
   def uniform_random(self, x_min, x_max):
      """Return 1D sample from scaled uniform random variable."""
      if x_min >= x_max:
         raise ValueError("Incorrect random number range.")

      x_range = x_max - x_min
      return (np.random.rand()*x_range) + x_min
         

      