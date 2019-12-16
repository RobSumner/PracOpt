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
   length - Storage length of the archive. Default value is 20.  

   Public Methods
   -------------- 
   add      - Add values to the archive. x is a sample point and f is objective
               function value at this point. 
   results  - Return results as an numpy array that can be used for plotting. 
               Each row of array is (x_point, f). 
   """

   def __init__(self, length=20):
      """Initialise the optimiser."""
      self.all_x_values = []
      self.all_obj_values = []
      self.L_length = length # Length of L dissimilarity archive.
      self.L_x_values = [] # L dissimilarity x values
      self.L_f_values = [] # L dissimilarity f values
      self.D_min = 0.1  # Minimum dissimilarity for all elements
      self.D_sim = 0.01 # Minimum dissimilarity to worst element
   
   def add(self, x, f):
      """Store current values in storage of all solutions and employ
         Best L-dissimilarity archiving scheme."""
      if not isinstance(x, np.ndarray):
         x = np.array([x])
      self.all_x_values.append(x)
      self.all_obj_values.append(f)

      if len(self.L_x_values) > 0:
         # Worst and Best current solution indices
         i_worst = np.argmax(self.L_f_values)
         i_best = np.argmin(self.L_f_values)
         # Most similar points
         lowest_sim_val, i_closest = self.most_similar(x)
 
      if len(self.L_x_values) == 0:
         # Always Store first element
         self.L_x_values.append(x)
         self.L_f_values.append(f)
         return
         
      elif len(self.L_x_values) < self.L_length and lowest_sim_val > self.D_min:
         # Store if archive not full and point is sufficiently 
         # dissimilar to all current entries.
         self.L_x_values.append(x)
         self.L_f_values.append(f)
         return
      elif len(self.L_x_values) == self.L_length:
         # Archive if full, dissimilar and better than one - replace worst.
         if lowest_sim_val > self.D_min and self.L_f_values[i_worst] > f:
            self.L_x_values[i_worst] = x
            self.L_f_values[i_worst] = f
            return

         # Archive if full, not dissimilar but best - replace most similar
         elif lowest_sim_val < self.D_min and self.L_f_values[i_best] > f:
            self.L_x_values[i_closest] = x
            self.L_f_values[i_closest] = f
            return

         # Archive if full, not dissimilar, better than most similar 
         # replace most similar.
         elif lowest_sim_val < self.D_sim and self.L_f_values[i_closest] > f:
            self.L_x_values[i_closest] = x
            self.L_f_values[i_closest] = f
            return
      return

   def results(self):
      """Convert stored data into a usable format."""
      n = len(self.all_x_values)
      d = len(self.all_x_values[0])
      x_data = np.reshape(np.array(self.all_x_values), (n,d))
      f_data = np.reshape(np.array(self.all_obj_values), (n,1))
      return np.concatenate((x_data, f_data), axis=1)

   def most_similar(self, point):
      """Find the point which is most similar in the archive.
         Returns the similarity value and index of the point in the archive."""
      if len(self.L_x_values) == 0:
         return None, None

      min_sim = 10e10
      index = 0
      for i, point_2 in enumerate(self.L_x_values):
         sim = self.similarity(point, point_2)
         if  sim < min_sim:
            min_sim = sim
            index = i
      return min_sim, index

   def similarity(self, point_1, point_2):
      """Find the similarity between two points, based upon a l2-norm 
         similarity measure."""
      # Convert into numpy arrays if not currently.
      if not isinstance(point_1, np.ndarray): point_1 = np.array([point_1])
      if not isinstance(point_2, np.ndarray): point_2 = np.array([point_2])

      # Check sizes match
      if point_1.shape != point_2.shape:
         raise ValueError('Size of points being compared for similarity do not match.')

      return np.linalg.norm(point_1 - point_2, 2)