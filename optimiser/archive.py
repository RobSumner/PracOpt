"""Store optimisation resuls

Classes
-------
Archive - Class to act as archive or storage method for optimsation.
"""

import numpy as np

class Archive:
   """Storage method for optimisation. 
   Parameters
   ----------

   Public Methods
   -------------- 
   add      - Add values to the archive. x is a sample point and f is objective
               function value at this point. 
   results  - Return results as an numpy array that can be used for plotting. 
               Each row of array is (x_point, f). 
   """

   def __init__(self):
      """Initialise the optimiser."""
      self.x_values = []
      self.obj_values = []
   
   def add(self, x, f):
      """Store the values in the archive."""
      if not isinstance(x, np.ndarray):
         x = np.array([x])
      self.x_values.append(x)
      self.obj_values.append(f)

   def results(self):
      """Convert stored data into a usable format."""
      n = len(self.x_values)
      d = len(self.x_values[0])
      x_data = np.reshape(np.array(self.x_values), (n,d))
      f_data = np.reshape(np.array(self.obj_values), (n,1))
      return np.concatenate((x_data, f_data), axis=1)