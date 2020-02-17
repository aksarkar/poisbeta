import poisbeta

from fixtures import *

def test_fit_poisson_beta_moment(simulate_easy):
  x, *theta = simulate_easy
  theta_hat = poisbeta.fit_poisson_beta_moment(x)
  assert np.isfinite(theta_hat).all()
  assert np.isclose(theta_hat, theta, rtol=0, atol=0.2).all()
