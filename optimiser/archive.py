"""Store optimisation resuls

Classes
-------
Archive - Class to act as archive or storage method for optimsation.
"""

import numpy as np
import time

class Archive:
   """Storage method for optimisation. 
   Parameters
   ----------
   length - Storage length of the archive. Default value is 20.  

   Public Methods
   -------------- 
   add - Add values to the archive. x is a sample point and f is objective
         function value at this point. 
   results - Return dissimilarity archive and all sample results as an numpy array 
             that can be used for plotting. 
   objective_data - For argument max_iter, return a [max_iter x 3] array of 
                    interpolated f and time values for linearly spaced 
                    iterations in range 1:1:max_iter.
                  - This can be used for comparison between methods. 
   reset - Reset variable parameters in the archive.
   most_similar - Identify the archive element most similar to point.
                - Returns similarity, index
                - Note a small similarity measure indicates similar points. 
   similarity - Returns the l2 norm similarity measure for two points.
   """

   def __init__(self, length=20):
      """Initialise the optimiser.
         Parameters:
         length - The length L of the L dissimilarity archive."""
      self.all_x_values = [] # Sample points
      self.all_f_values = []  # Objective function values.
      self.all_time_track = [] # Iteration & time of archiving
      self.L_length = length # Length of L dissimilarity archive.
      self.L_x_values = [] # L dissimilarity x values
      self.L_f_values = [] # L dissimilarity f values
      self.D_min = 0.25  # Minimum dissimilarity for all elements
      self.D_sim = 0.025 # Minimum dissimilarity to worst element
   
   def add(self, x, f, iteration):
      """Store current values in storage of all solutions and employ
         Best L-dissimilarity archiving scheme.
         Parameters:
         x - Current x sample point. 
         f - Current objective function value. 
         iteration - Current evaluation iteration."""
      if not isinstance(x, np.ndarray):
         x = np.array([x])
      self.all_x_values.append(x)
      self.all_f_values.append(f)
      self.all_time_track.append(np.array([iteration, time.process_time()]))

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
      """Convert stored data into a usable format.
         Returns:
         L_samples - From L dissimilarity archive, one row per sample.
                   - Row:[sample point, objective value]
         all_samples - Every sample recorded, one row per sample.
                     - Row:[sample point, objective value, evaluation, time]
                     - Evaluation Iteration may not increase evenly as it is 
                        the iteration of objective function evaluation."""
      n = len(self.all_x_values)
      d = len(self.all_x_values[0])
      x_data = np.reshape(np.array(self.all_x_values), (n,d))
      f_data = np.reshape(np.array(self.all_f_values), (n,1))
      time = np.reshape(np.array(self.all_time_track), (n,2))
      all_samples = np.concatenate((x_data, f_data, time), axis=1)


      x_data = np.reshape(np.array(self.L_x_values), (len(self.L_x_values),d))
      f_data = np.reshape(np.array(self.L_f_values), (len(self.L_f_values),1))
      L_samples = np.concatenate((x_data, f_data), axis=1)

      return L_samples, all_samples

   def objective_data(self, max_iter):
      """Function to summarise objective function data in a usable form.
         This can be used to compare between methods. 
         Parameters:
         max_iter - Number of iterations that data should be to extrapolates
                    or interpolated to fit.
         Returns:
         data - A [max_iter x 3] array of interpolated objective function
                values during the current search. 
              - Array row = [evaluation iteration, time, function value]."""
      n = len(self.all_x_values)
      f_data = np.reshape(np.array(self.all_f_values), (n,1))
      time = np.reshape(np.array(self.all_time_track), (n,2))

      # Rescale so first sample is at time 0
      time[:,1] -= time[0,1]

      # Store minimum function value at given iteration
      lowest_f = np.zeros((n,))
      min_f = 10e20
      for i, f in enumerate(f_data):
         if f < min_f:
            min_f = f
         lowest_f[i] = min_f

      # Linearly interpolate to give values for all evaluation iterations
      new_evals = np.linspace(1, max_iter, max_iter)
      new_f = np.interp(new_evals, time[:,0], lowest_f)
      new_wall_time = np.interp(new_evals, time[:,0], time[:,1])

      return np.concatenate( ( np.reshape(new_evals, (max_iter,1)), 
                               np.reshape(new_wall_time, (max_iter,1)),
                               np.reshape(new_f, (max_iter,1)) ), axis=1)

   def reset(self):
      """Reset variable parameters."""
      self.all_x_values = []
      self.all_f_values = []
      self.all_time_track = []
      self.L_x_values = [] 
      self.L_f_values = [] 

   def most_similar(self, point):
      """Find the point which is most similar in the archive.
         Parameters:
         point - Sample point to compare elements of the archive to. 
         Returns:
         min_sim - The lowest similarity value found in the archive.
         index - Index of the archive element with this similarity value."""
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
         similarity measure.
         Parameters:
         point_1, point_2 - Two points to be compared. 
                          - Order does not matter.
         Returns:
         similarity - l2 norm of vector between points."""
      # Convert into numpy arrays if not currently.
      if not isinstance(point_1, np.ndarray): point_1 = np.array([point_1])
      if not isinstance(point_2, np.ndarray): point_2 = np.array([point_2])

      # Check sizes match
      if point_1.shape != point_2.shape:
         raise ValueError('Size of points being compared for similarity do not match.')

      return np.linalg.norm(point_1 - point_2, 2)