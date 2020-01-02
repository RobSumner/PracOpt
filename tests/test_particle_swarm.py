"""Test the particle swarm optimisation class"""

import pytest 
import numpy as np
import copy
from optimiser.optimiser import ParticleSwarm
from optimiser.objective import Shubert, ObjectiveTest
from optimiser.utils import evaluate

# PSO with test objective functions
@pytest.fixture
def new_test_pso():
   """Return a new instance of the simulated annealing class
      with 1D test function.
      Don't run optimise on this as objective is constant - will hang."""
   obj = ObjectiveTest()
   pso = ParticleSwarm(obj)
   pso.max_evaluations = 10
   return pso

# PSO classes with 5D Shubert objective
@pytest.fixture
def new_pso5():
   """Return a new instance of the simulated annealing class
      with 2D Shubert objective function."""
   obj = Shubert(5)
   return ParticleSwarm(obj)

# Test functions
def test_pso_init(new_test_pso):
   """Test init of pso."""
   assert new_test_pso.dimension == 1
   assert new_test_pso.n_particles == 25
   assert new_test_pso.particle_x.shape == (25, 1)
   assert new_test_pso.particle_v.shape == (25, 1)

   assert new_test_pso.particle_best_x.shape == (25, 1)
   assert new_test_pso.particle_best_f.shape == (25, 1)
   assert new_test_pso.global_best_x.shape == (1,)
   assert new_test_pso.global_best_f == np.Inf

def test_pso_rand(new_pso5, new_test_pso):
   """Test random number generator."""
   # Fix seed for reproducible results
   np.random.seed(seed=1)
   with pytest.raises(ValueError):
      # low bigger than high.
      new_pso5.uniform_random(1,-1)

   for _ in range(10):
      sample = new_pso5.uniform_random(-1,1)
      assert sample >= -1
      assert sample <= 1

   samps = 200
   samples = np.zeros((samps,1))
   for i in range (samps):
      samples[i] = new_test_pso.uniform_random(-2,2)
   assert round(np.mean(samples)*100)/100 == 0.02
   assert round(np.std(samples)*100)/100 == 1.22

   # Test positive sample mean
   samples = samples[samples > 0]
   assert round(np.mean(samples)*100)/100 == 1

def test_pso_particle_init(new_pso5):
   """Test particle initialisation for PSO."""
   new_pso5.initialise()

   for i in range(new_pso5.n_particles):
      pos = new_pso5.particle_x[i,:]
      # Best value should be f(current position)
      assert new_pso5.particle_best_f[i] == new_pso5.objective.f(pos)
      # Global best value
      assert new_pso5.global_best_f <= new_pso5.particle_best_f[i]
      # Best position should be current position
      assert all([a == b for a, b in zip(pos, new_pso5.particle_best_x[i,:])])
      # Velocity should be in range [-4,4]
      assert all([a <= b for a, b in zip(new_pso5.particle_v[i,:], 
                  4*np.ones((new_pso5.dimension,1)))])
      
def test_velocity_update(new_pso5):
   """Test updating the velocity of a particle."""
   # Index outside range will fail
   with pytest.raises(ValueError):
      new_pso5.update_velocity(-1)
   with pytest.raises(ValueError):
      new_pso5.update_velocity(26)

   # Check an updated velocity value
   i = 10
   new_pso5.initialise()
   current_v = copy.deepcopy(new_pso5.particle_v[i,:])
   new_pso5.update_velocity(i)
   new_v = new_pso5.particle_v[i,:]
   for i in range(new_pso5.dimension):
      assert current_v[i] != new_v[i]

def test_run_reset(new_pso5):
   """Test running the optimisation."""
   np.random.seed(seed=1)
   new_pso5.max_evaluations = 200
   new_pso5.run()
   assert pytest.approx(new_pso5.global_best_f, -40.81)

   new_pso5.reset()
   assert new_pso5.global_best_f == np.Inf
   assert new_pso5.objective.evaluations == 0

