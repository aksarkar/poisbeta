import numpy as np
import pytest

def _simulate(n, kon=None, koff=None, kr=None, seed=None):
  if seed is not None:
    np.random.seed(seed)
  if kon is None:
    kon = np.random.lognormal(sigma=0.5)
  if koff is None:
    koff = np.random.lognormal(sigma=0.5)
  if kr is None:
    kr = np.random.lognormal(mu=3, sigma=0.5)
  p = np.random.beta(a=kon, b=koff, size=n)
  x = np.random.poisson(kr * p)
  return x, np.log(kr), np.log(kon), np.log(koff)

@pytest.fixture
def simulate_easy():
  return _simulate(n=1000, kon=1, koff=1, kr=32, seed=0)
