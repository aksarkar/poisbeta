import numpy as np
import poisbeta.mcmc
import pytest
import scipy.stats as st

from fixtures import *

@pytest.mark.xfail
def test__slice_sample():
  f = st.norm().logpdf
  samples = [poisbeta.mcmc._slice_sample(f, 0, width=4) for i in range(1000)]
  d, p = st.kstest(samples, 'norm')
  assert p >= 0.05

def test__conf_logpdf_p():
  grid = np.linspace(0, 1, 1000)
  f = poisbeta.mcmc._cond_logpdf_p(p=grid, x=10, kr=5, kon=.5, koff=.5)
  assert np.isfinite(f).all()

def test__conf_logpdf_kr():
  grid = np.logspace(-3, 1, 1000)
  f = poisbeta.mcmc._cond_logpdf_kr(kr=grid, x=10, p=.5, ar=10, br=1)
  assert np.isfinite(f).all()

def test__conf_logpdf_kon():
  grid = np.logspace(-3, 1, 1000)
  f = poisbeta.mcmc._cond_logpdf_kon(kon=grid, p=.5, koff=.5, aon=1, bon=100)
  assert np.isfinite(f).all()

def test__conf_logpdf_koff():
  grid = np.logspace(-3, 1, 1000)
  f = poisbeta.mcmc._cond_logpdf_koff(koff=grid, p=.5, kon=.5, aoff=1, boff=100)
  assert np.isfinite(f).all()

def test_fit_pois_beta_mcmc_one_sample(simulate_easy):
  x, *theta = simulate_easy
  samples = poisbeta.mcmc.fit_poisson_beta_mcmc(x, n_samples=1, ar=1, br=x.max(), aon=1, bon=100, aoff=1, boff=100)
  assert samples.shape == (1, 3)

def test_fit_pois_beta_mcmc_n_samples(simulate_easy):
  x, *theta = simulate_easy
  n_samples = 2
  samples = poisbeta.mcmc.fit_poisson_beta_mcmc(x, n_samples=n_samples, ar=1, br=x.max(), aon=1, bon=100, aoff=1, boff=100)
  assert samples.shape == (n_samples, 3)
