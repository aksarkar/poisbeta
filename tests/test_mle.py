import poisbeta
import poisbeta.mle

from fixtures import *

def test_poisson_beta_logpmf(simulate_easy):
  x, *theta = simulate_easy
  theta = np.array(theta)
  log_px = poisbeta.mle.poisson_beta_logpmf(theta, x, s=1)
  assert np.isfinite(log_px).all()

def test_poisson_beta_pmf(simulate_easy):
  x, *theta = simulate_easy
  theta = np.array(theta)
  px = poisbeta.mle.poisson_beta_pmf(theta, x, s=1)
  assert np.isfinite(px).all()
  assert (px >= 0).all()
  assert (px <= 1).all()

def test_poisson_beta_pmf_norm():
  theta = np.array([5, 0, 0])
  px = poisbeta.mle.poisson_beta_pmf(theta, np.arange(0, 200, 1), s=1)
  assert np.isclose(px.sum(), 1)

def test_fit_poisson_beta_mle(simulate_easy):
  x, *theta = simulate_easy
  theta_hat = poisbeta.fit_poisson_beta_mle(x)
  assert np.isfinite(theta_hat).all()
  assert np.isclose(theta_hat, theta, rtol=0, atol=0.2).all()

def test_fit_poisson_beta_mle_oracle_init(simulate_easy):
  x, *theta = simulate_easy
  theta_hat = poisbeta.fit_poisson_beta_mle(x, init=theta)
  assert np.isfinite(theta_hat).all()
  assert np.isclose(theta_hat, theta, rtol=0, atol=0.2).all()

def test_fit_poisson_beta_mle_random_init(simulate_easy):
  x, *theta = simulate_easy
  theta_hat = poisbeta.fit_poisson_beta_mle(x, init=np.random.normal(size=3))
  assert np.isfinite(theta_hat).all()
  assert np.isclose(theta_hat, theta, rtol=0, atol=0.2).all()

def test_fit_poisson_beta_mle_oracle_init_oracle_llik(simulate_easy):
  x, *theta = simulate_easy
  theta = np.array(theta)
  oracle_llik = poisbeta.mle.poisson_beta_logpmf(theta, x, s=1).sum()
  theta_hat = poisbeta.fit_poisson_beta_mle(x, init=theta)
  llik = poisbeta.mle.poisson_beta_logpmf(theta_hat, x, s=1).sum()
  assert llik >= oracle_llik
