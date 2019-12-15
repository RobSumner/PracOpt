"""Test the archive class."""

import pytest
import numpy as np
from optimiser.archive import Archive

@pytest.fixture
def new_arch():
   """Return an archive populated with some results."""
   arch = Archive()
   for i in range(100):
      arch.add(np.array([i]), 100)
   return arch

def test_archive_init():
   """Test archive initialisation"""
   arch = Archive()
   assert len(arch.obj_values) == 0
   assert len(arch.x_values) == 0

def test_archive_store(new_arch):
   """Test the storage methods for archive."""
   assert len(new_arch.obj_values) == 100
   assert len(new_arch.x_values) == 100

   new_arch.add(-1,-1)
   assert len(new_arch.obj_values) == 101
   assert len(new_arch.x_values) == 101

def test_archive_results(new_arch):
   """Test the results method for archive."""
   results = new_arch.results()
   shape = results.shape
   print(results)
   assert shape[0] == 100
   assert shape[1] == 2
