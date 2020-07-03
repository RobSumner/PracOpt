"""Test the archive class."""

import pytest
import numpy as np

from pracopt.archive import Archive

@pytest.fixture
def new_arch():
   """Return an archive populated with constant f results."""
   arch = Archive()
   arch._D_min = 0.1
   arch._D_sim = 0.01
   for i in range(100):
      arch.add(np.array([i]), 100, i)
   return arch

@pytest.fixture
def new_arch2():
   """Return an archive populated with non constant f results."""
   arch = Archive()
   arch._D_min = 0.1
   arch._D_sim = 0.01
   for i in range(100):
      arch.add(np.array([i]), i, i)
   return arch

@pytest.fixture
def new_arch3():
   """Return an archive with artificially populated L archive."""
   arch = Archive()
   arch._D_min = 0.1
   arch._D_sim = 0.01
   arch._L_x_values = [np.array([1,1]), np.array([1,2]), np.array([3,1])]
   arch._L_f_values = [1, 2, 3]
   return arch

def test_archive_init():
   """Test archive initialisation"""
   arch = Archive()
   assert len(arch._all_f_values) == 0
   assert len(arch._all_x_values) == 0
   assert len(arch._all_time_track) == 0
   assert len(arch._L_x_values) == 0
   assert len(arch._L_f_values) == 0

def test_archive_add(new_arch, new_arch2):
   """Test the storage methods for archive."""
   assert len(new_arch._all_f_values) == 100
   assert len(new_arch._all_time_track) == 100
   assert len(new_arch._all_x_values) == 100

   new_arch.add(-1,-1,-1)
   assert len(new_arch._all_f_values) == 101
   assert len(new_arch._all_time_track) == 101
   assert len(new_arch._all_x_values) == 101

   # Test the L dissimilarity archive
   # Basic adding of numbers 1:100 - 1:20 will be stored
   assert len(new_arch2._L_f_values) == new_arch2._L_length
   f_vals = np.linspace(0, new_arch2._L_length, new_arch2._L_length-1)
   assert sum(new_arch2._L_f_values) == sum(f_vals)

   # Test adding values which then change
   # For example if arrays are edited, this saves values
   # rather than pointers.
   x_point = np.ones((1,))*5
   vals = np.ones((2,))*2
   new_arch2.add(x_point, vals[0], vals[1])
   assert new_arch2._all_x_values[-1] == 5
   assert new_arch2._all_f_values[-1] == 2
   x_point[0] = 6
   vals[0] = 3
   assert new_arch2._all_x_values[-1] == 5
   assert new_arch2._all_f_values[-1] == 2

def test_archive_results(new_arch):
   """Test the results method for archive."""
   l, samples = new_arch.results()
   shape1 = samples.shape
   assert shape1[0] == 100
   assert shape1[1] == 4
   shape2 = l.shape
   assert shape2[0] == 20
   assert shape2[1] == 2

def test_archive_objective_data(new_arch):
   """Test method for interpolating function values."""
   data = new_arch.objective_data(110)
   shape = data.shape
   assert shape[0] == 110
   assert shape[1] == 3
   assert sum(data[:,2]) == 110*100
   assert data[-1,0] == 110
   assert data[0,1] == pytest.approx(0, abs=1e04)

def test_archive_reset(new_arch2):
   """Test the reset method for archive."""
   assert len(new_arch2._all_x_values) == 100
   assert len(new_arch2._all_f_values) == 100
   assert len(new_arch2._all_time_track) == 100
   assert len(new_arch2._L_x_values) == 20
   assert len(new_arch2._L_f_values) == 20

   new_arch2.reset()

   assert len(new_arch2._all_x_values) == 0
   assert len(new_arch2._all_f_values) == 0
   assert len(new_arch2._all_time_track) == 0
   assert len(new_arch2._L_x_values) == 0
   assert len(new_arch2._L_f_values) == 0


@pytest.mark.parametrize("point_1, point_2, value",
      [([1], [1], 0),
       ([1, 1], [1, 1], 0),
       ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 0),
       ([1, 2], [1, 1], 1),
       ([5, -2], [-6, 4], 12.529964)])

def test_similarity_measure(new_arch, point_1, point_2, value):
   """Test the similarity measure values."""
   with pytest.raises(ValueError):
      new_arch.similarity([1], [1,1])
   assert new_arch.similarity(point_1, point_2) == pytest.approx(value)


@pytest.mark.parametrize("point, sim, index",
      [(np.array([1,1]), 0, 0),
       (np.array([1,2]), 0, 1),
       (np.array([3,1]), 0, 2),
       (np.array([2,1]), 1, 0),
       (np.array([3,3]), 2, 2)
       ])

def test_most_similar(new_arch3, point, sim, index):
   """Test function returning most similar point."""
   sim_val, index_val = new_arch3.most_similar(point)
   assert sim_val == sim
   assert index_val == index


