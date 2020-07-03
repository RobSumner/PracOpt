"""Objective Functions

This module contains a set of objective function definitions.

"""

from abc import ABC, abstractmethod
import numpy as np

class Objective(ABC):
    """Base Objective Class.

    Base objective class defining required objective function pattern
    for optimiser.

    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def is_feasible(self, x):
        """Check feasibility of point.

        Check the feasibility of a sample point x.

        Parameters
        ----------
        x : :class:`numpy.array`
            Sample point to check.

        Returns
        -------
        feasible : :obj:`bool`
            True if point is feasible, False otherwise.

        """
        pass

    @abstractmethod
    def f(self, x):
        """"Objective function.

        Calculate objective function value at a sample point.

        Parameters
        ----------
        x : :class:`numpy.array`
            Sample point to check.

        Returns
        -------
        obj_val : :obj:`float`
            Objective function value at sample point.

        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the objective function.

        """
        pass

class Shubert(Objective):
	"""N-dimensional Shubert function.

    Class implementing an N-dimensional Shubert function.

    Parameters
	----------
	n : :obj:`int`
        Dimension of the function.

    Attributes
    ----------
    evaluations : :obj:`int`
        The number of objective function evaluation calls made.

	"""

	def __init__(self, n):
		if type(n) is not int or n <= 0:
			raise ValueError('Dimension must be integer greater than 0.')
		self.n = n

		self._x_min = -2
		self._x_max = 2

		self.evaluations = 0

	def is_feasible(self, x, index=None):
		"""Check feasibility of point.

        Check the feasibility of a sample point x.

        Parameters
        ----------
        x : :class:`numpy.array`
            Sample point to check.

        Returns
        -------
        feasible : :obj:`bool`
            True if point is feasible, False otherwise.

        """
		size = 1
		for dim in np.shape(x): size *= dim
		if len(x) != self.n or size != self.n:
			raise ValueError('Dimension of x does not match function definition.')
		if index is not None:
			if index < 0 or index >= self.n:
				raise ValueError('Index exceeds dimension of function.')

		if index is None:
			for x_i in x:
				if x_i < self._x_min or x_i > self._x_max:
					return False
		elif x[index] < self._x_min or x[index] > self._x_max:
				return False

		return True

	def f(self, x):
		""""Objective function.

        Calculate objective function value at a sample point.

        Parameters
        ----------
        x : :class:`numpy.array`
            Sample point to check.

        Returns
        -------
        obj_val : :obj:`float`
            Objective function value at sample point.

        """
		if not self.is_feasible(x):
			raise ValueError("Point x must lie in feasible region.")

		self.evaluations += 1
		obj_val = 0
		for i in range(0, self.n):
			for j in range(1, 6):
				obj_val += j*np.sin((j+1)*x[i] + j)
		return obj_val

	def reset(self):
		"""Reset the objective function.

        """
		self.evaluations = 0

class ObjectiveTest(Objective):
	"""1 dimensional test function with limits [-2,2]

    This objective function returns the absolute value of x.

    Attributes
    ----------
    evaluations : :obj:`int`
        The number of objective function evaluation calls made.

	"""

	def __init__(self):
		self.n = 1
		self._x_max = 2
		self._x_min = -2
		self._max_step = 1

	def is_feasible(self, x, index=None):
		"""Check feasibility of point.

        Check the feasibility of a sample point x.

        Parameters
        ----------
        x : :class:`numpy.array`
            Sample point to check.

        Returns
        -------
        feasible : :obj:`bool`
            True for all points for this test case.

        """
		for x_i in x:
			if x_i < self._x_min or x_i > self._x_max:
				return False
		return True

	def f(self, x):
		""""Objective function.

        Objective function is absolute value of x at the give point.

        Parameters
        ----------
        x : :class:`numpy.array`
            Sample point to check.

        Returns
        -------
        obj_val : :obj:`float`
            Absolute value of x.

        """
		return np.linalg.norm(x, 1)

	def reset(self):
		"""Reset the objective function.

        """
		self.evaluations = 0

