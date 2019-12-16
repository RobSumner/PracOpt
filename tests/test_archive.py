"""Test the archive class."""

import pytest
import numpy as np
from optimiser.archive import Archive

@pytest.fixture
def new_arch():
   """Return an archive populated with constant f results."""
   arch = Archive()
   for i in range(100):
      arch.add(np.array([i]), 100)
   return arch

@pytest.fixture
def new_arch2():
   """Return an archive populated with non constant f results."""
   arch = Archive()
   for i in range(100):
      arch.add(np.array([i]), i)
   return arch

@pytest.fixture
def new_arch3():
   """Return an archive with artificially populated L archive."""
   arch = Archive()
   arch.L_x_values = [np.array([1,1]), np.array([1,2]), np.array([3,1])]
   arch.L_f_values = [1, 2, 3]
   return arch

def test_archive_init():
   """Test archive initialisation"""
   arch = Archive()
   assert len(arch.all_obj_values) == 0
   assert len(arch.all_x_values) == 0

def test_archive_add(new_arch, new_arch2, new_arch4):
   """Test the storage methods for archive."""
   assert len(new_arch.all_obj_values) == 100
   assert len(new_arch.all_x_values) == 100

   new_arch.add(-1,-1)
   assert len(new_arch.all_obj_values) == 101
   assert len(new_arch.all_x_values) == 101

   # Test the L dissimilarity archive
   # Basic adding of numbers 1:100 - 1:20 will be stored
   assert len(new_arch2.L_f_values) == new_arch2.L_length
   f_vals = np.linspace(0, new_arch2.L_length, new_arch2.L_length-1)
   assert sum(new_arch2.L_f_values) == sum(f_vals)

def test_archive_results(new_arch):
   """Test the results method for archive."""
   results = new_arch.results()
   shape = results.shape
   print(results)
   assert shape[0] == 100
   assert shape[1] == 2


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

   
