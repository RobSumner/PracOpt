"""Perform Optimisation

Classes
-------
SimAnneal - Class to perform Simulated Annealing optimisation

"""

from optimiser import objective

class SimAnneal:
   """Perform Simulated Annealing Optimisiation
   Parameters
   ----------
   objective   - Objective class describing function to be optimised.

   Public Methods
   -------------- 
   """

   def __init__(self, objective):
      """Initialise the optimiser."""
      self.objective = objective
      self.iteration = 0

