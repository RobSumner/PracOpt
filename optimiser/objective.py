"""Define Objective Function

Classes
-------
Shubert - n dimensional Shubert function
"""

import numpy as np

class Shubert:
   """n dimensional Shubert function
   Parameters
   ----------
   n    - Dimension of the function. 

   Public Methods
   --------------
   is_feasible(x) - Returns True if point x is feasible or False if not. 
   f(x)  - Returns value of objective function at point x. 
   """

   def __init__(self, n):
      """Initialise Shubert function properties."""
      if type(n) is not int or n <= 0:
         raise ValueError('Dimension must be integer greater than 0.')
      self.n = n

      self.x_min = -2
      self.x_max = 2

   def is_feasible(self, x, index=None):
      """Returns true if point x is feasible.
         If index is supplied, the coordinate with this
         index is checked for feasibility."""
      if len(x) != self.n:
         raise ValueError('Dimension of x does not match function definition.')
      if index is not None:
         if index < 0 or index >= self.n:
            raise ValueError('Index exceeds dimension of function.')

      if index is None:
         for x_i in x:
            if x_i < self.x_min or x_i > self.x_max:
               return False
      elif x[index] < self.x_min or x[index] > self.x_max:
            return False
      return True

   def f(self, x):
      """Return value of Shubert function at point x."""
      if not self.is_feasible(x):
         raise ValueError("Point x must lie in feaisble region.")

      obj_val = 0
      for i in range(0, self.n):
         for j in range(1, 6):
            obj_val += j*np.sin((j+1)*x[i] + j)
      return obj_val
   
