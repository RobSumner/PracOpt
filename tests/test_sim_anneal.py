"""Test the simulated annealing optimiser."""

import pytest
from optimiser.optimiser import SimAnneal
from optimiser.objective import Shubert

@pytest.fixture
def new_sim2():
   """Return a new instance of the simulated annealing class
      with 2D Shubert objective function.
   """
   obj = Shubert(2)
   return SimAnneal(obj)

@pytest.fixture
def new_sim5():
   """Return a new instance of the simulated annealing class
      with 2D Shubert objective function.
   """
   obj = Shubert(5)
   return SimAnneal(obj)

def test_sim_anneal_init(new_sim2, new_sim5):
   """Test initialisation of the simulated annealing class."""
   assert type(new_sim2.objective) == Shubert
   assert new_sim2.iteration == 0

   assert type(new_sim5.objective) == Shubert
   assert new_sim5.iteration == 0



