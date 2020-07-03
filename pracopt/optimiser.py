"""Optimisation methods.

This module contains a number of optimisation methods (implemented as
classes).

"""

import copy
from enum import Enum

from abc import ABC, abstractmethod
import numpy as np

from pracopt import objective
from pracopt.archive import Archive
from pracopt.utils import progress_bar

class Optimiser(ABC):
    """Optimiser base class.

    Base abstract optimiser class.

    Parameters
    ----------
    objective : :class:`pracopt.objective.Objective`
        Objective function being optimised.

    Attributes
    ----------
    archive : :class:`pracopt.achive.Archive`
        Archiving class for optimiser.

    dimension : :obj:`int`
        Dimension of the objective function being optimised.

    """

    def __init__(self, objective):
        super().__init__()
        self.objective = objective
        self.dimension = objective.n
        self.archive = Archive()

    @abstractmethod
    def run(self):
        """Run optimisation.

        Run the optimisation defined by this class.

        """
        pass

    @abstractmethod
    def reset(self):
        """Reset optimiser.

        Reset the optimiser.

        """
        pass

    def _uniform_random(self, x_min=None, x_max=None, dim=1):
        """Uniform random sample.

        Return nD sample from scaled uniform random variable.

        Parameters
        ----------
        x_min, x_max : :obj:`float`, optional.
            The range for generated numbers. This defaults to the provided
            objective function range if None is provided.

        dim :obj:`int`, optional.
            Dimension of random sample to be generated. Default is 1.

        Returns
        -------
        x : :class:`numpy.array`
            Randomly generated sample of dimension dim.
        """

        if x_min is None:
            x_min = self.objective._x_min
        if x_max is None:
            x_max = self.objective._x_max

        if x_min >= x_max: raise ValueError("Incorrect random number range.")

        x_range = x_max - x_min
        return (np.random.rand(dim,1)*x_range) + x_min


class SimAnneal(Optimiser):
    """Simulated Annealing Optimiser

    Optimiser class implementing simulated annealing.

    Parameters
    ----------
    objective : :class:`pracopt.objective.Objective`
        Objective function being optimised.

    trial_mode : {'basic','vanderbilt', 'parks'}, optional.
        The trial proposal mode to be used. Default value is 'vanderbilt'.

    initial_temp_mode : {'preset', 'kirkpatrick', 'white'}, optional.
        The method used to select the initial temperature. Default value
        is 'kirkpatrick'.

    max_step : :obj:`float`
        Value define neighbourhood in which trial solutions can be found.

    Attributes
    ----------
    archive : :class:`pracopt.achive.Archive`
        Archiving class for optimiser.

    objective : :class:`pracopt.objective.Objective`
        Objective function being optimised.

    x : :class:`numpy.array`
        The current sample point.

    """

    _TRIAL_MODES = ['basic','vanderbilt', 'parks']
    _TEMP_MODES = ['preset', 'kirkpatrick', 'white']
    _alpha = 0.1
    _omega = 2.1
    _decrement_length = 100
    _max_evaluations = 10000

    def __init__(self, objective, trial_mode="vanderbilt",
                        initial_temp_mode="kirkpatrick", max_step=2):
        super().__init__(objective)

        assert trial_mode in self._TRIAL_MODES
        self.trial_mode = trial_mode
        assert initial_temp_mode in self._TEMP_MODES
        self._initial_temp_mode = initial_temp_mode

        self.max_step = max_step

        self.reset()

    def run(self):
        """Run the optimisation.

        Run simulated annealing on the objective function until
        self._max_evaluations is reached.

        """

        self._set_initial_temp()

        self.x = self._uniform_random(dim=self.dimension)

        while self.objective.evaluations < self._max_evaluations:
            progress_bar(self.objective.evaluations, self._max_evaluations)

            f0 = self.objective.f(self.x)
            x1 = self._new_trial_solution()
            f1 = self.objective.f(x1)
            while not self._acceptable_solution(f1 - f0):
                x1 = self._new_trial_solution()
                f1 = self.objective.f(x1)
            self.x = x1

            self.archive.add(x1, f1, self.objective.evaluations)
            self._update_temperature()

        # Reset output from carriage return
        print("")

    def reset(self):
        """Reset optimiser.

        Reset the optimiser annealing schedule and storate for new trial
        solutions.

        """
        self.x = np.zeros((self.dimension, 1))

        # Annealing schedule
        self._initial_T = 10e10
        self._temp_start_id = 0
        self._current_T = 10e10
        self._trials = 0
        self._acceptances = 0

        self.archive.reset()
        self.objective.reset()

        # Initialise Q  and D matrices to diagonal of maximum allowable step.
        self._Q_matrix = np.eye(self.dimension) * self.max_step
        self._D_matrix = np.eye(self.dimension) * self.max_step

    def _acceptable_solution(self, df):
        """Test if solution is acceptable

        Test if a solution is acceptable based upon the change in objective
        function.

        This applies the simulated annealing acceptance criteria:
            * Decreases in objective function are always accepted.
            * Increases in objective function are accepted randomly with a
              chance given by the simulated annealing acceptance
              probability.

        Parameters
        ----------
        df : :obj:`float`
            The change in objective function.

        Returns
        -------
        accept : :obj:`bool`
            True if solution should be accepted, False otherwise.

        """
        if df < 0:
            self._acceptances += 1
            return True

        else:
            p_accept = np.exp(-1*df/self._current_T)
            sample = self._uniform_random(0,1)
            if sample <= p_accept:
                self._acceptances += 1
                return True
            else:
                return False

    def _new_trial_solution(self, x0=None):
        """Produce new trial solution.

        Produce a new trial solution based upon the current new trial
        solution mode set at initialisation.
            * basic - Sample a new point from a range of allowable points.
            * vanderbilt - Apply Vanderbilt and Louie sampling (1984)
                which includes correlations of points in sampling.
            * parks - Apply Parks sampling (1990).

        Parameters
        ----------
        x0 : :class:`numpy.array`, optional.
            Starting point for new trial. If None provided, the current
            self.x value is used.

        Returns
        -------
        x_new : :class:`numpy.array`
            New trial solution.

        """
        if x0 is None:
            x0 = self.x

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
                x_min = max(self.objective._x_min, x0[i] - self.max_step)
                x_max = min(self.objective._x_max, x0[i] + self.max_step)
                x_new[i] = self._uniform_random(x_min=x_min, x_max=x_max)

                # Check that new value in dimension only is feasible
                while not self.objective.is_feasible(x_new, i):
                    x_new[i] = self._uniform_random(x_min=x_min, x_max=x_max)

        # Vanderbilt and Louie method [1984]
        elif self.trial_mode == "vanderbilt":
            # Calculate covariance matrix of path at current temperature
            n = len(self.archive._all_x_values) - (self._temp_start_id)
            if n > 1:
                # Covariance of all values at this temperature
                path = np.reshape(np.array(self.archive._all_x_values[self._temp_start_id:]),
                                                    (n,self.dimension))
                path_cov = np.cov(path.T)
            else:
                path_cov = np.eye(self.dimension)

            # Update generator matrix
            S_matrix = self._Q_matrix*self._Q_matrix.T
            S_matrix = (1 - self._alpha)*S_matrix + self._alpha*self._omega*path_cov
            self._Q_matrix = np.linalg.cholesky(S_matrix)

            # Set maximum step size by normalising Q_matrix
            max_row_norm_Q = max(np.linalg.norm(self._Q_matrix, 1, axis=1))
            Q_mat = self._Q_matrix * self.max_step / (np.sqrt(3) * max_row_norm_Q)

            # Sample until feasible
            u = self._uniform_random(x_min=-np.sqrt(3), x_max=np.sqrt(3), dim=self.dimension)
            x_new = x0 + np.matmul(Q_mat, u)
            while not self.objective.is_feasible(x_new):
                u = self._uniform_random(x_min=-np.sqrt(3), x_max=np.sqrt(3), dim=self.dimension)
                x_new = x0 + np.matmul(Q_mat, u)

        # Parks method [1990]
        elif self.trial_mode == 'parks':
            #  Set maximum step size by normalizing D matrix
            max_row_norm_D = max(np.linalg.norm(self._D_matrix, 1, axis=1))
            D_mat = self._D_matrix * self.max_step / max_row_norm_D

            # Sample until feasible
            u = self._uniform_random(x_min=-1, x_max=1, dim=self.dimension)
            x_new = x0 + np.matmul(D_mat, u)
            while not self.objective.is_feasible(x_new):
                u = self._uniform_random(x_min=-1, x_max=1, dim=self.dimension)
                x_new = x0 + np.matmul(D_mat, u)

            # Update D matrix
            R_matrix = np.eye(self.dimension)
            mag_change = np.abs(x_new - x0)
            for i in range(self.dimension): R_matrix[i][i] = mag_change[i][0]
            self._D_matrix = (1 - self._alpha)*self._D_matrix + self._alpha*self._omega*R_matrix

        self._trials += 1
        return x_new

    def _update_temperature(self):
        """Update the annealing temperature.

        Update annealing temperature following the annealing schedule.
            * Temperature is updated every decrement_length evaluations.
            * Q matrix is updated whenever the temperature is decreased.

        """
        # Implement basic length based decrement.
        self._current_T = self._initial_T*0.95**(self._trials // self._decrement_length)

        # Reset Q matrix whenever temperature is lowered.
        if (self._trials % self._decrement_length) == 0:
            self._Q_matrix = np.eye(self.dimension)*self.max_step
            self._temp_start_id = len(self.archive._all_x_values)
        return

    def _set_initial_temp(self):
        """Set the initial temperature

        Set the initial annealing temperature based upon initial_temp_mode:
            *kirkpatric - Based on Kirkpatrick [1984] where T0 set so av.
                            prob. of solution increasing f is ~0.8.
            * white - Based on White [1984] where T0 set = sigma where
                        sigma is std. dev. of variation in objective
                        function during initial search.
            * preset - Use a constant value preset in the class. Default
                        value is 10e10 in this case.

        """
        if self._initial_temp_mode == "preset":
            # Temp already set in class - no calculation needed.
            return

        # Calculate T0 following White/Kirkpatrick
        samples = 100
        df_samples = np.zeros((samples,1))
        for i in range(samples):
            x0 = self._uniform_random(dim=self.dimension)
            f0 = self.objective.f(x0)

            x1 = self._new_trial_solution(x0)
            f1 = self.objective.f(x1)

            for _ in range(10):
                if f1 <= f0:
                    x1 = self._new_trial_solution(x0)
                    f1 = self.objective.f(x1)
                else:
                    df_samples[i] = f1 - f0

        # Extract zeros from array (failed trials)
        df_samples = df_samples[df_samples != 0]

        if self._initial_temp_mode == "kirkpatrick":
            self._initial_T = -1*np.mean(df_samples)/np.log(0.8)
        elif self._initial_temp_mode == "white":
            self._initial_T = np.std(df_samples)

        self._current_T = self._initial_T

        self.objective.reset()


class ParticleSwarm(Optimiser):
    """Particle Swarm Optimiser

    Optimiser class implementing Particle Swarm Optimisation.

    Parameters
    ----------
    objective : :class:`pracopt.objective.Objective`
        Objective function being optimised.

    n_particles : :obj:`int`
        Number of particles to use.

    omega : :obj:`float`
        Interia term for particle velocity in velocity update.

    phi_p : :obj:`float`
        Intertia term for best particle position in velocity update.

    phi_g : :obj:`float`
        Intertia term for best global position in velocity update.

    Attributes
    ----------
    archive : :class:`pracopt.achive.Archive`
        Archiving class for optimiser.

    objective : :class:`pracopt.objective.Objective`
        Objective function being optimised.

    """

    _max_evaluations = 10000

    def __init__(self, objective, n_particles=25, omega=0.5, phi_p=0.25, phi_g=0.75):
        super().__init__(objective)

        self._n_particles = n_particles

        assert omega >= 0 and phi_g >= 0 and phi_p >= 0
        assert (phi_p + phi_g) <= 1
        self._omega = omega
        self._phi_p = phi_p
        self._phi_g = phi_g

        self.reset()

    def run(self):
        """Run the optimisation.

        Run Particle Swarm optimisation on the objective function until
        self._max_evaluations is reached.

        """
        self._initialise_particles()

        while self.objective.evaluations < self._max_evaluations:
            progress_bar(self.objective.evaluations, self._max_evaluations)

            for particle_id in range(self._n_particles):
                self._update_velocity(particle_id)

                self._particle_x[particle_id,:] += \
                    self._particle_v[particle_id,:]

                f_val = self.objective.f(self._particle_x[particle_id,:])
                if f_val < self._particle_best_f[particle_id]:
                    self._particle_best_f[particle_id] = f_val
                    self._particle_best_x[particle_id,:] = \
                        self._particle_x[particle_id,:]

                if f_val < self._global_best_f:
                    self._global_best_f = f_val
                    self._global_best_x = self._particle_x[particle_id,:]

                self.archive.add(
                    self._particle_x[particle_id,:], f_val,
                    self.objective.evaluations
                )

        # Reset output from carriage return
        print("")

    def reset(self):
        """Reset optimiser.

        Reset the optimiser annealing schedule and storate for new trial
        solutions.

        """
        self._particle_x = np.zeros((self._n_particles, self.dimension))
        self._particle_v = np.zeros((self._n_particles, self.dimension))
        self._particle_best_x = np.zeros((self._n_particles, self.dimension))
        self._particle_best_f = np.ones((self._n_particles, 1)) * np.Inf
        self._global_best_x = np.zeros((self.dimension))
        self._global_best_f = np.Inf

        self.archive.reset()
        self.objective.reset()

    def _initialise_particles(self):
        """Initialise particles

        Initialise particle positions and velocities. Positions and velocities
        are initialised to cover the full range of the objective function.

        """
        for i in range(self._n_particles):
            new_x = np.reshape(self._uniform_random(
                dim=self.dimension), (self.dimension)
            )
            self._particle_x[i,:] = new_x

            self._particle_best_x[i,:] = new_x
            f_val = self.objective.f(new_x)
            self._particle_best_f[i,0] = f_val
            self.archive.add(
                self._particle_x[i,:], f_val, self.objective.evaluations
            )

            if f_val < self._global_best_f:
                self._global_best_f = f_val
                self._global_best_x = new_x

            v_width = self.objective._x_max - self.objective._x_min
            new_v = np.reshape(self._uniform_random(
                -1*v_width, v_width,dim=self.dimension), (self.dimension)
            )
            self._particle_v[i,:] = new_v

    def _update_velocity(self, particle_index):
        """Update particle velocity

        Update the velocity (in place) of particle with index particle_index.

        Note
        ----
        Feasibility is achieved by iteratively reducing the size of the
        inertia term. This enforces feasibility as phi_p + phi_g <= 1 so when
        inertia term becomes small, the convex hull of remaining velocity
        terms lies inside feasible region.

        Parameters
        ----------
        particle_index : :obj:`int`
            Index of particle to update.

        """
        assert particle_index >= 0 and particle_index < self._n_particles

        current_x = self._particle_x[particle_index,:]
        current_v = self._particle_v[particle_index,:]
        current_p = self._particle_best_x[particle_index,:]

        v_new = np.zeros((self.dimension))
        for d in range(self.dimension):
            # Differenctes between particle and global best
            diff_p = current_p[d] - current_x[d]
            diff_g = self._global_best_x[d] - current_x[d]

            # Random selections for [rp, rg]
            rand = self._uniform_random(x_min=0, x_max=1, dim=2)

            v_new[d] = self._omega * current_v[d] + \
                         self._phi_p * rand[0] * diff_p + \
                         self._phi_g * rand[1] * diff_g

            # Reduce inertia until new position is feasible.
            # When inertia = 0, position must be feasible as it lies
            # in convex hull of two feasible points.
            x_new = current_x + np.ones(current_x.shape) * v_new[d]
            it = 1
            while not self.objective.is_feasible(x_new, index=d):
                v_new[d] = self._omega*0.9**it * current_v[d] + \
                         self._phi_p * rand[0] * diff_p + \
                         self._phi_g * rand[1] * diff_g
                x_new = current_x + np.ones(current_x.shape) * v_new[d]
                it += 1

        self._particle_v[particle_index] = v_new
        return


