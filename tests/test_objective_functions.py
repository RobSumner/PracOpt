"""Test the objective functions."""

import pytest
from optimiser import objective

def test_check():
   A = 1
   assert A == 1

@pytest.fixture
def new_shubert5():
    """Return a new instance of the Shubert objective with n = 5."""
    return objective.Shubert(5)

@pytest.fixture
def new_shubert2():
    """Return a new instance of the Shubert objective with n = 5."""
    return objective.Shubert(2)

def test_shubert_init():
   """Test the initialisation of the Shubert function."""
   obj1 = objective.Shubert(1)
   assert obj1.n == 1
   assert obj1.x_min == -2
   assert obj1.x_max == 2

   with pytest.raises(ValueError):
      objective.Shubert(0)
   
   with pytest.raises(ValueError):
      objective.Shubert(-10)

@pytest.mark.parametrize("point,feasible", [([1,1],      True),  
                                            ([2,2],      True), 
                                            ([-2,-2],    True),  
                                            ([-3,1],     False),
                                            ([-1,3],     False), 
                                            ([100,100],  False)])
def test_shubert2_feasible(new_shubert2, point, feasible):
   """Test the feasibility check for 2D Shubert function."""
   assert new_shubert2.is_feasible(point) == feasible
   with pytest.raises(ValueError):
      new_shubert2.is_feasible([1,1,1])

@pytest.mark.parametrize("point,feasible", [([1,1,1,1,1],            True), 
                                            ([2,2,2,2,2],            True), 
                                            ([-2,-2,-2,-2,-2],       True),
                                            ([-3,1,1,1,1],           False),
                                            ([-1,3,2,3,4],           False), 
                                            ([100,100,100,100,100],  False)])
def test_shubert5_feasible(new_shubert5, point, feasible):
   """Test the feasibility check for 5D Shubert function."""
   assert new_shubert5.is_feasible(point) == feasible
   with pytest.raises(ValueError):
      new_shubert5.is_feasible([1,1,1])


@pytest.mark.parametrize("point,value", [([1,1],    -6.3145), 
                                         ([-1,1],  -15.7793),
                                         ([1,-1],  -15.7793), 
                                         ([1.5,2],  -0.4637),
                                         ([2,2],    -5.6495), 
                                         ([-2,-2],   4.1640)])
def test_shubert2_values(new_shubert2, point, value):
   """Test the returned objective function values for 2D Shubert function."""
   # Check against MATLAB 4dp values
   obj_val = round(new_shubert2.f(point)*10e03)/10e03
   assert obj_val == value
   with pytest.raises(ValueError):
      new_shubert2.f([1,2,3])

@pytest.mark.parametrize("point,value", [([1,1,1,1,1],          -15.7862), 
                                         ([-1,1,-1,1,-1],       -44.1807),
                                         ([1,-1,1,-1,1],        -34.7159), 
                                         ([1.5,-1,1.5,-1,1],    -23.6793),
                                         ([1.5,2,1.5,-2,1],       0.8220), 
                                         ([-0.5,1.9,1.5,-2,1.2], 12.2095)])
def test_shubert5_values(new_shubert5, point, value):
   """Test the returned objective function values for 5D Shubert function."""
   # Check against MATLAB 4dp values
   obj_val = round(new_shubert5.f(point)*10e03)/10e03
   assert obj_val == value

   with pytest.raises(ValueError):
      new_shubert5.f([1,2,3])


