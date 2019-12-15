"""Test the simulated annealing optimiser."""

import pytest
import numpy as np
from optimiser.optimiser import SimAnneal, TrialMode, InitialTempMode
from optimiser.objective import Shubert, ObjectiveTest

@pytest.fixture
def new_sim2():
   """Return a new instance of the simulated annealing class
      with 2D Shubert objective function.
   """
   obj = Shubert(2)
   return SimAnneal(obj, TrialMode.BASIC, InitialTempMode.KIRKPATRICK)

@pytest.fixture
def new_sim5():
   """Return a new instance of the simulated annealing class
      with 2D Shubert objective function.
   """
   obj = Shubert(5)
   return SimAnneal(obj, TrialMode.BASIC, InitialTempMode.KIRKPATRICK)

@pytest.fixture
def new_test_sim():
   """Return a new instance of the simulated annealing class
      with 5D test function.
   """
   obj = ObjectiveTest()
   return SimAnneal(obj, TrialMode.BASIC, InitialTempMode.KIRKPATRICK)

def test_sim_anneal_init(new_sim2, new_sim5):
   """Test initialisation of the simulated annealing class."""
   assert type(new_sim2.objective) == Shubert
   assert new_sim2.trial_mode == TrialMode.BASIC

   assert type(new_sim5.objective) == Shubert
   assert new_sim5.trial_mode == TrialMode.BASIC

def test_sim_anneal_rand(new_sim5, new_test_sim):
   """Test random number generator."""
   # Fix seed for reproducible results
   np.random.seed(seed=1)
   with pytest.raises(ValueError):
      new_sim5.uniform_random(1,-1)

   for _ in range(10):
      sample = new_sim5.uniform_random(-1,1)
      assert sample >= -1
      assert sample <= 1

   samps = 200
   samples = np.zeros((samps,1))
   for i in range (samps):
      samples[i] = new_test_sim.uniform_random(-2,2)
   assert round(np.mean(samples)*100)/100 == 0.02
   assert round(np.std(samples)*100)/100 == 1.22

   # Test positive sample mean
   samples = samples[samples > 0]
   assert round(np.mean(samples)*100)/100 == 1

def test_sim_anneal_new_trial_solution(new_sim5):
   """Test new trial proposal method."""
   # Check class 
   assert all([a == b for a, b in zip(new_sim5.x, np.zeros((5,1)))])
   # Test starting point using class start point. 
   new_x = new_sim5.new_trial_solution()
   assert all([a != b for a, b in zip(new_x, np.zeros((5,1)))])

   # Test using provided start point
   new_x = new_sim5.new_trial_solution()

def test_initial_temperature(new_test_sim, new_sim2):
   """Test initial temperature calculation."""
   np.random.seed(seed=1)

   # Kirkpatrick method
   assert new_test_sim.current_T == 10000
   new_test_sim.initial_temp()
   assert round(new_test_sim.current_T*100)/100 == 2.56

   # White method
   new_test_sim.initial_temp_mode = InitialTempMode.WHITE
   new_test_sim.initial_temp()
   assert round(new_test_sim.current_T*100)/100 == 0.44
   
   # Kirkpatric method
   assert new_sim2.current_T == 10000
   new_sim2.initial_temp()
   assert round(new_sim2.current_T/10)*10 == 30

   # White method
   new_sim2.initial_temp_mode = InitialTempMode.WHITE
   new_sim2.initial_temp()
   assert round(new_sim2.current_T) == 8


