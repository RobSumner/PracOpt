"""Archive optimisation results.

This module contains tools for archiving optimisation results. Archiving
can improve the quality of the overall solution by tracking progress of
global optimisation methods.

"""

import time
import copy

import numpy as np
import scipy.io

class Archive:
    """Archive Solutions

    Archive optimisation results using best L-Dissimilarity measure.

    Parameters
    ----------
    length : :obj:`int`, optional.
        Storage length of the dissimilarity archive.
        Default value is 20.

    """

    def __init__(self, length=20):
        self._all_x_values = [] # All Sample points provided to archive.
        self._all_f_values = []  # All Objective function values.
        self._all_time_track = [] # Iteration & time of archiving
        self._L_length = length # Length of L dissimilarity archive.
        self._L_x_values = [] # L dissimilarity x values
        self._L_f_values = [] # L dissimilarity f values
        self._D_min = 0.25  # Minimum dissimilarity for all elements
        self._D_sim = 0.025 # Minimum dissimilarity to worst element

    def add(self, x, f, iteration):
        """Add solution to archive

        Add a new value to the archive and apply dissimilarity method
        to update archive.

        Parameters
        ----------

        x : :obj:`float`
            Current sample point.

        f : :obj:`float`
            Current objective function value.

        iteration : :obj:`int`
            Current evaluation iteration.

        """
        # Deepcopy to prevent alteration.
        x = copy.deepcopy(x)
        f = copy.deepcopy(f)
        iteration = copy.deepcopy(iteration)

        if not isinstance(x, np.ndarray):
            x = np.array([x])

        # Store every value in list.
        self._all_x_values.append(x)
        self._all_f_values.append(f)
        self._all_time_track.append(np.array([iteration, time.process_time()]))

        # Update L-dissimilarity archive - only stores smaller number.
        if len(self._L_x_values) > 0:
            # Worst and Best current solution indices
            i_worst = np.argmax(self._L_f_values)
            i_best = np.argmin(self._L_f_values)
            # Most similar points
            lowest_sim_val, i_closest = self.most_similar(x)

        if len(self._L_x_values) == 0:
            # Always Store first element
            self._L_x_values.append(x)
            self._L_f_values.append(f)
            return

        elif len(self._L_x_values) < self._L_length and lowest_sim_val > self._D_min:
            # Store if archive not full and point is sufficiently
            # dissimilar to all current entries.
            self._L_x_values.append(x)
            self._L_f_values.append(f)
            return
        elif len(self._L_x_values) == self._L_length:
            # Archive if full, dissimilar and better than one - replace worst.
            if lowest_sim_val > self._D_min and self._L_f_values[i_worst] > f:
                self._L_x_values[i_worst] = x
                self._L_f_values[i_worst] = f
                return

            # Archive if full, not dissimilar but best - replace most similar
            elif lowest_sim_val < self._D_min and self._L_f_values[i_best] > f:
                self._L_x_values[i_closest] = x
                self._L_f_values[i_closest] = f
                return

            # Archive if full, not dissimilar, better than most similar
            # replace most similar.
            elif lowest_sim_val < self._D_sim and self._L_f_values[i_closest] > f:
                self._L_x_values[i_closest] = x
                self._L_f_values[i_closest] = f
                return
        return

    def results(self):
        """Return results

        Return the archive results.

        Returns
        -------
        L_samples : :class:`numpy.array`
            The current contents of the archive, one row per entry.
            Each row is [sample, objective value].

        all_samples : :class:`numpy.array`
            Every sample recorded, one row per sample
            Each row is [sample point, objective value,
                         evaluation iteration, time]
            Evaluation Iteration may not increase evenly as it is the
            iteration of objective function evaluation.

        """
        n = len(self._all_x_values)
        d = len(self._all_x_values[0])
        x_data = np.reshape(np.array(self._all_x_values), (n,d))
        f_data = np.reshape(np.array(self._all_f_values), (n,1))
        time = np.reshape(np.array(self._all_time_track), (n,2))
        all_samples = np.concatenate((x_data, f_data, time), axis=1)

        x_data = np.reshape(np.array(self._L_x_values), (len(self._L_x_values),d))
        f_data = np.reshape(np.array(self._L_f_values), (len(self._L_f_values),1))
        L_samples = np.concatenate((x_data, f_data), axis=1)

        return L_samples, all_samples

    def objective_data(self, max_iter):
        """Summarise objective function data.

        Summarise objective function data in a usable form which can be
        used to compare between methods.

        Parameters
        ----------
        max_iter : :obj:`int`
            Number of iterations that data should be extrapolated or
            interpolated to fit.

        Returns
        -------
        data : :class:`numpy.array`
            A [max_iter x 3] array of interpolated objective function
            values during the current search.
            Each row is [evaluation iteration, time, function value].

        """
        n = len(self._all_x_values)
        f_data = np.reshape(np.array(self._all_f_values), (n,1))
        time = np.reshape(np.array(self._all_time_track), (n,2))

        # Rescale so first sample is at time 0
        time[:,1] -= time[0,1]

        # Store minimum function value at given iteration
        lowest_f = np.zeros((n,))
        min_f = 10e20
        for i, f in enumerate(f_data):
            if f < min_f:
                min_f = f
            lowest_f[i] = min_f

        # Interpolate to give values for all evaluation iterations
        new_evals = np.linspace(1, max_iter, max_iter)
        new_f = np.interp(new_evals, time[:,0], lowest_f)
        new_wall_time = np.interp(new_evals, time[:,0], time[:,1])

        return np.concatenate( ( np.reshape(new_evals, (max_iter,1)),
                                np.reshape(new_wall_time, (max_iter,1)),
                                np.reshape(new_f, (max_iter,1)) ), axis=1)

    def reset(self):
        """Reset archive.

        Reset the archive and empty all storage. This should be called
        at the end of an optimisation run.

        """
        self._all_x_values = []
        self._all_f_values = []
        self._all_time_track = []
        self._L_x_values = []
        self._L_f_values = []

    def most_similar(self, point):
        """Find most similar point.

        Find the point which is most similar in the archive.

        Parameters
        ----------
        point : :class:`numpy.array`
            Sample point to compare elements of the archive to.

        Returns
        -------
        min_sim : :obj:`float`
            The lowest similarity value found in the archive.

        index : :obj:`int`
            Index of the archive element with lowest similarity value.

        """
        if len(self._L_x_values) == 0:
            return None, None

        min_sim = 10e10
        index = 0
        for i, point_2 in enumerate(self._L_x_values):
            sim = self.similarity(point, point_2)
            if  sim < min_sim:
                min_sim = sim
                index = i

        return min_sim, index

    def similarity(self, point_1, point_2):
        """Similarity between two points.

        Calculate similarity measure (using l2-norm) between two points.

        Parameters
        ----------
        point_1, point_2 : :class:`numpy.array`
            Two points to be compared. Note, point order does not matter.

        Returns
        -------
        similarity : :obj:`float`
            Similarity measure (L2 norm) between points.

        """
        if not isinstance(point_1, np.ndarray): point_1 = np.array([point_1])
        if not isinstance(point_2, np.ndarray): point_2 = np.array([point_2])

        if point_1.shape != point_2.shape:
            msg = 'Size of points being compared for similarity do not match.'
            raise ValueError(msg)

        return np.linalg.norm(point_1 - point_2, 2)

    def save_mat(self, filepath):
        """Save archive results.

        Save archive results as a .mat file for use in MATLAB.

        Parameters
        ----------
        filepath : :obj:`string`
            The path of the file to save results to.
            Does not need to have .mat extension on end.

        """
        L_samples, all_samples = self.results()
        results_dict = {'L_samples':L_samples,
                        'all_samples':all_samples}
        scipy.io.savemat(filepath + '.mat', results_dict)

        return