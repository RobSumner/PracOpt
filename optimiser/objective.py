"""Define Objective Function

Classes
-------
Shubert - n dimensional Shubert function.
ObjectiveTest - n dimensional test function. 
"""

import numpy as np

class Shubert:
   """n dimensional Shubert function
   Parameters
   ----------
   n - Dimension of the function. 

   Public Methods
   --------------
   is_feasible - Returns True if point x is feasible or False if not. 
   f - Returns value of objective function at point x. 
   reset - Reset variable parameters. 
   """

   def __init__(self, n):
      """Initialise Shubert function properties.
         Parameters:
         n - Dimension of the function."""
      if type(n) is not int or n <= 0:
         raise ValueError('Dimension must be integer greater than 0.')
      self.n = n

      self.x_min = -2
      self.x_max = 2
      self.evaluations = 0

   def is_feasible(self, x, index=None):
      """Check feasibility of a point. 
         Parameters:
         x - Point to be checked.
         index - If supplied, the specific component x[index] is checked only.
               - If None supplied (default) the whole vector x is checked.
         Returns:
         Bool - True if feasible, False if not."""
      # Check the input variable
      size = 1
      for dim in np.shape(x): size *= dim
      if len(x) != self.n or size != self.n:
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
      """Evaluate objective function at point.
         Parameters:
         x - Point to evaluate function at.
         Returns:
         Float - Value of objective function at point."""
      if not self.is_feasible(x):
         raise ValueError("Point x must lie in feaisble region.")

      # Increment evaluation number
      self.evaluations += 1

      obj_val = 0
      for i in range(0, self.n):
         for j in range(1, 6):
            obj_val += j*np.sin((j+1)*x[i] + j)
      return obj_val

   def reset(self):
      """Reset variable objective function parameters."""
      self.evaluations = 0

class ObjectiveTest:
   """1 dimensional test function with limits [-2,2]
   Parameters
   ---------- 

   Public Methods
   --------------
   is_feasible(x) - Returns True if x lies in range [-2,2].
   f(x)  - Returns magnitude of x. 
   """

   def __init__(self):
      """Initialise Shubert function properties."""
      self.n = 1
      self.x_max = 2
      self.x_min = -2

   def is_feasible(self, x, index=None):
      """Returns true for all points."""
      for x_i in x:
         if x_i < self.x_min or x_i > self.x_max:
            return False
      return True

   def f(self, x):
      """Return absolute value of x."""
      return np.linalg.norm(x, 1)
   
   def reset(self):
      """Reset variable objective function parameters."""
      self.evaluations = 0
   
