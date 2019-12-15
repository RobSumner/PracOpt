"""Test the simulated annealing optimiser."""

import pytest
import numpy as np
from optimiser.optimiser import SimAnneal, TrialMode, InitialTempMode
from optimiser.objective import Shubert, ObjectiveTest

@pytest.fixture
def new_test_sim():
   """Return a new instance of the simulated annealing class
      with 5D test function.
   """
   obj = ObjectiveTest()
   return SimAnneal(obj, TrialMode.BASIC, InitialTempMode.KIRKPATRICK)

@pytest.fixture
def new_test_sim_white():
   """Return a new instance of the simulated annealing class
      with 5D test function.
   """
   obj = ObjectiveTest()
   return SimAnneal(obj, TrialMode.BASIC, InitialTempMode.WHITE)

@pytest.fixture
def new_sim2():
   """Return a new instance of the simulated annealing class
      with 2D Shubert objective function.
   """
   obj = Shubert(2)
   return SimAnneal(obj, TrialMode.BASIC, InitialTempMode.KIRKPATRICK)

@pytest.fixture
def new_sim2_white():
   """Return a new instance of the simulated annealing class
      with 2D Shubert objective function.
   """
   obj = Shubert(2)
   return SimAnneal(obj, TrialMode.BASIC, InitialTempMode.WHITE)

@pytest.fixture
def new_sim5():
   """Return a new instance of the simulated annealing class
      with 2D Shubert objective function.
   """
   obj = Shubert(5)
   return SimAnneal(obj, TrialMode.BASIC, InitialTempMode.KIRKPATRICK)

@pytest.fixture
def new_sim5_white():
   """Return a new instance of the simulated annealing class
      with 2D Shubert objective function.
   """
   obj = Shubert(5)
   return SimAnneal(obj, TrialMode.BASIC, InitialTempMode.WHITE)

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
   assert new_sim5.trials == 0
   new_x = new_sim5.new_trial_solution()
   assert all([a != b for a, b in zip(new_x, np.zeros((5,1)))])
   assert new_sim5.trials == 1

   # Test using provided start point
   new_x = new_sim5.new_trial_solution([0,0,0,0,0])
   assert new_sim5.trials == 2

def test_set_initial_temperature(new_test_sim, new_test_sim_white, new_sim2, 
                                 new_sim2_white, new_sim5, new_sim5_white):
   """Test initial temperature calculation."""
   np.random.seed(seed=1)

   # Kirkpatrick method
   assert new_test_sim.initial_T == 10e10
   assert new_test_sim.current_T == 10e10
   new_test_sim.set_initial_temp()
   assert round(new_test_sim.initial_T*100)/100 == 2.43

   # White method
   assert new_test_sim_white.current_T == 10e10
   assert new_test_sim_white.current_T == 10e10
   new_test_sim_white.set_initial_temp()
   assert round(new_test_sim_white.initial_T*100)/100 == 0.39
   
   # Kirkpatric method
   assert new_sim2.initial_T == 10e10
   assert new_sim2.current_T == 10e10
   new_sim2.set_initial_temp()
   assert round(new_sim2.initial_T/10)*10 == 40
   assert new_sim2.initial_T == new_sim2.current_T

   # White method
   assert new_sim2_white.initial_T == 10e10
   assert new_sim2_white.current_T == 10e10
   new_sim2_white.set_initial_temp()
   assert round(new_sim2_white.initial_T) == 6
   assert new_sim2_white.initial_T == new_sim2_white.current_T

   # Commented for speed - fairly slow to run.
   # Kirkpatric method
   assert new_sim5.initial_T == 10e10
   assert new_sim5.current_T == 10e10
   new_sim5.set_initial_temp()
   assert round(new_sim5.initial_T/10)*10 == 50
   assert new_sim5.initial_T == new_sim5.current_T

   # # White method
   assert new_sim5_white.initial_T == 10e10
   assert new_sim5_white.current_T == 10e10
   new_sim5_white.set_initial_temp()
   assert round(new_sim5_white.initial_T) == 11
   assert new_sim5_white.initial_T == new_sim5_white.current_T

def test_acceptable_solution(new_sim2, new_sim5):
   """Test the acceptable solution method."""
   # Negative solutions always accepted
   assert new_sim2.acceptable_solution(-1)
   assert new_sim5.acceptable_solution(-1)

   # Very large values not accepter
   assert not new_sim2.acceptable_solution(10e20)
   assert not new_sim5.acceptable_solution(10e20)

   # Give p = 0.5 with T = 10000
   sum_1 = 0
   sum_2 = 0
   runs = 10000
   for _ in range(runs):
      if new_sim2.acceptable_solution(6.93147e10):
         sum_1 += 1
      if new_sim5.acceptable_solution(6.93147e10):
         sum_2 += 1
   assert round(sum_1/runs*10)/10 == 0.5
   assert round(new_sim2.acceptances/1000)*1000 == 5000
   assert round(sum_2/runs*10)/10 == 0.5
   assert round(new_sim5.acceptances/1000)*1000 == 5000

def test_temperature_update(new_sim2):
   """Test temperature decrement."""
   assert new_sim2.current_T == 10e10
   new_sim2.update_temperature()
   assert new_sim2.current_T == 10e10

   for _ in range(new_sim2.decrement_length):
      new_sim2.new_trial_solution([0,0])
   new_sim2.update_temperature()
   assert new_sim2.current_T == 9.5e10

def test_run(new_sim2):
   """Test the run function for 200 evaluations."""
   np.random.seed(seed=1)
   new_sim2.max_evaluations = 200
   new_sim2.run()
   assert round(new_sim2.initial_T) == 30
   assert round(new_sim2.current_T) == 25

   


