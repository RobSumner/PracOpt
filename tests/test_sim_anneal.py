"""Test the simulated annealing optimiser."""

import pytest
import numpy as np
from optimiser.optimiser import SimAnneal, TrialMode
from optimiser.objective import Shubert

@pytest.fixture
def new_sim2():
   """Return a new instance of the simulated annealing class
      with 2D Shubert objective function.
   """
   obj = Shubert(2)
   return SimAnneal(obj, TrialMode.BASIC)

@pytest.fixture
def new_sim5():
   """Return a new instance of the simulated annealing class
      with 2D Shubert objective function.
   """
   obj = Shubert(5)
   return SimAnneal(obj, TrialMode.BASIC)

def test_sim_anneal_init(new_sim2, new_sim5):
   """Test initialisation of the simulated annealing class."""
   assert type(new_sim2.objective) == Shubert
   assert new_sim2.iteration == 0
   assert new_sim2.trial_mode == TrialMode.BASIC

   assert type(new_sim5.objective) == Shubert
   assert new_sim5.iteration == 0
   assert new_sim5.trial_mode == TrialMode.BASIC

def test_sim_anneal_rand(new_sim5):
   """Test random number generator."""
   with pytest.raises(ValueError):
      new_sim5.uniform_random(1,-1)

   for _ in range(10):
      sample = new_sim5.uniform_random(-1,1)
      assert sample >= -1
      assert sample <= 1

def test_sim_anneal_new_trial(new_sim5):
   assert all([a == b for a, b in zip(new_sim5.x, np.zeros((5,1)))])
   
