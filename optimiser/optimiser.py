"""Implement optimisation methods

Classes
-------
SimAnneal - Class to perform Simulated Annealing optimisation

Functions
---------
progress_bar - Creates a progress bar for use in terminals when running opts.

evaluate - Function to evaluate performance of an optimisation method. 
           This performs multiple runs to account for stochastic methods. 

"""

from optimiser import objective
from optimiser.archive import Archive
from enum import Enum
import numpy as np 
import copy

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

   def __init__(self, objective, trial_mode="vanderbilt", 
                  initial_temp_mode="kirkpatrick", max_step=2):
      """Initialise the optimiser.
         Parameters:
         trial_mode - Defines how the trial solutions are generated.
         initial_temp_mode - Defines how initial temperature is set.
         max_step - Maximum allowable step size to be used in the objective function. 
      """
      # Set new trial suggestion method
      trial_modes = ['basic','vanderbilt', 'parks']
      if trial_mode not in trial_modes:
         raise ValueError('Trial Mode not recognised.')
      self.trial_mode = trial_mode

      # Set initial temperature method
      initial_temp_modes = ['preset', 'kirkpatrick', 'white']
      if initial_temp_mode not in initial_temp_modes:
         raise ValueError('Initial Temperature Mode not recognised.')
      self.initial_temp_mode = initial_temp_mode

      # Set objective function
      self.objective = objective
      self.dimension = objective.n
      self.max_step = max_step

      # Point, Objective value & change in objective.
      self.x = np.zeros((self.dimension, 1)) # Sample point

      # Trial solution method data
      self.alpha = 0.1
      self.omega = 2.1

      # Initialise Q & D to diagonal of maximum allowable step. 
      self.Q_matrix = np.eye(self.dimension)*self.max_step
      self.D_matrix = np.eye(self.dimension)*self.max_step

      # Annealing schedule
      self.initial_T = 10e10
      self.current_T = 10e10
      self.temp_start_id = 0 # Id of first sample at new temperature.
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
      self.temp_start_id = 0
      self.current_T = 10e10
      self.trials = 0 # Total length of chain
      self.acceptances = 0 # Total number of acceptances

      # Reset archive & objective
      self.archive.reset()
      self.objective.reset()
      
      # Initialise Q  and D matrices to diagonal of maximum allowable step. 
      self.Q_matrix = np.eye(self.dimension)*self.max_step
      self.D_matrix = np.eye(self.dimension)*self.max_step

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
      # x0 must be numpy array of correct size
      if not isinstance(x0, np.ndarray):
         x0 = np.array(x0)
      try:
         x0 = np.reshape(np.array(x0), (self.dimension, 1))
      except ValueError:
         raise ValueError('Incorrect starting point shape for new trial.')
         
      # Simple diagonal (C) matrix update.
      x_new = np.zeros((self.dimension,1))
      if self.trial_mode == "basic":
         # Sample new feasible position from altered range
         # This avoids wasted samples and result is the same.
         for i in range(self.dimension):
            x_min = max(self.objective.x_min, x0[i] - self.max_step)
            x_max = min(self.objective.x_max, x0[i] + self.max_step)
            x_new[i] = self.uniform_random(x_min=x_min, x_max=x_max)
            
            # Check that new value in dimension only is feasible
            while not self.objective.is_feasible(x_new, i):
               x_new[i] = self.uniform_random(x_min=x_min, x_max=x_max)

      # Vanderbilt and Louie method [1984]
      elif self.trial_mode == "vanderbilt":
         # Calculate covariance matrix of path at current temperature
         n = len(self.archive.all_x_values) - (self.temp_start_id)
         if n > 1:
            # Covariance of all values at this temperature
            path = np.reshape(np.array(self.archive.all_x_values[self.temp_start_id:]),
                                       (n,self.dimension))
            path_cov = np.cov(path.T)
         else:
            path_cov = np.eye(self.dimension)

         # Update generator matrix
         S_matrix = self.Q_matrix*self.Q_matrix.T
         S_matrix = (1 - self.alpha)*S_matrix + self.alpha*self.omega*path_cov
         self.Q_matrix = np.linalg.cholesky(S_matrix)

         # Set maximum step size by normalising Q_matrix
         max_row_norm_Q = max(np.linalg.norm(self.Q_matrix, 1, axis=1))
         Q_mat = self.Q_matrix * self.max_step / (np.sqrt(3) * max_row_norm_Q)
         
         # Sample until feasible
         u = self.uniform_random(x_min=-np.sqrt(3), x_max=np.sqrt(3), dim=self.dimension)
         x_new = x0 + np.matmul(Q_mat, u)
         while not self.objective.is_feasible(x_new):
            u = self.uniform_random(x_min=-np.sqrt(3), x_max=np.sqrt(3), dim=self.dimension)
            x_new = x0 + np.matmul(Q_mat, u)

      # Parks method [1990]
      elif self.trial_mode == 'parks':
         #  Set maximum step size by normalizing D matrix
         max_row_norm_D = max(np.linalg.norm(self.D_matrix, 1, axis=1))
         D_mat = self.D_matrix * self.max_step / max_row_norm_D
         
         # Sample until feasible
         u = self.uniform_random(x_min=-1, x_max=1, dim=self.dimension)
         x_new = x0 + np.matmul(D_mat, u)
         while not self.objective.is_feasible(x_new):
            u = self.uniform_random(x_min=-1, x_max=1, dim=self.dimension)
            x_new = x0 + np.matmul(D_mat, u)

         # Update D matrix
         R_matrix = np.eye(self.dimension)
         mag_change = np.abs(x_new - x0)
         for i in range(self.dimension): R_matrix[i][i] = mag_change[i][0]
         self.D_matrix = (1 - self.alpha)*self.D_matrix + self.alpha*self.omega*R_matrix

      self.trials += 1
      return x_new

   def update_temperature(self):
      """Update the annealing temperature."""
      # Implement basic length based decrement. 
      self.current_T = self.initial_T*0.95**(self.trials // self.decrement_length)

      # Reset Q matrix whenever temperature is lowered.
      if (self.trials % self.decrement_length) == 0:
         self.Q_matrix = np.eye(self.dimension)*self.max_step
         self.temp_start_id = len(self.archive.all_x_values)
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
      if self.initial_temp_mode == "preset":
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
      if self.initial_temp_mode == "kirkpatrick":
         # Set T to be average
         self.initial_T = -1*np.mean(df_samples)/np.log(0.8)
      elif self.initial_temp_mode == "white":
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
   