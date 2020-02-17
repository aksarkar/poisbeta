import numpy as np
import scipy.stats as st

def _slice_sample(f, init, width=0.1, max_steps=100, **kwargs):
  """Return samples from the density proportional to exp(f)

  f - log density (up to a constant)
  init - initial value
  kwargs - additional arguments to f

  """
  # Auxiliary variable defines the slice
  z = f(init, **kwargs) - np.random.exponential()
  # Step out
  left = init - width * np.random.uniform()
  right = left + width
  left_steps = int(np.random.uniform() * max_steps)
  for _ in range(left_steps):
    if z < f(left, **kwargs):
      break
    left -= width
  for _ in range(max_steps - left_steps):
    if z < f(right, **kwargs):
      break
    right += width
  # Step in
  while right > left:
    proposal = left + np.random.uniform() * (right - left)
    if z < f(proposal, **kwargs):
      return proposal
    elif proposal < init:
      left = proposal
    else:
      right = proposal
  raise RuntimeError('failed to find an acceptable sample')

def _cond_logpdf_p(p, x, kr, kon, koff):
  return (x + kon - 1) * np.log(p) + (koff - 1) * np.log(1 - p) - kr * p

def _cond_logpdf_kr(kr, x, p, ar, br):
  return ((x + ar - 1) * np.log(kr) - (p + br) * kr).sum()

def _cond_logpdf_kon(kon, p, koff, aon, bon):
  return (kon * np.log(p) + (aon - 1) * np.log(kon)
          - bon * kon - sp.gammaln(kon) + sp.gammaln(kon + koff)).sum()

def _cond_logpdf_koff(koff, p, on, aoff, boff):
  return (koff * np.log(1 - p) + (aoff - 1) * np.log(koff)
          - boff * koff - sp.gammaln(koff) + sp.gammaln(on + koff)).sum()

def fit_poisson_beta_mcmc(x, n_samples, ar, br, aon, bon, aoff, boff, verbose=False):
  """Return samples from the posterior p(kon, koff, kr | x)

  x - counts (n,)
  n_samples - number of samples to draw
  ar, br - prior parameters kr ~ Gamma(ar, br)
  aon, bon - prior parameters kon ~ Gamma(aon, bon)
  aoff, boff - prior parameters koff ~ Gamma(aoff, boff)

  """
  samples = []
  # Important: these are fixed
  Fr = st.gamma(a=ar, scale=1 / br)
  Fon = st.gamma(a=aon, scale=1 / bon)
  Foff = st.gamma(a=aoff, scale=1 / boff)
  # Draw initial sample from the prior
  kr = Fr.rvs(size=1)
  kon = Fon.rvs(size=1)
  koff = Foff.rvs(size=1)
  p = st.beta(a=kon, b=koff).rvs(size=x.shape)
  for t in range(n_samples):
    samples.append((kr, kon, koff))
    for i in range(x.shape[0]):
      p[i] = _slice_sample(_cond_logpdf_p, init=p[i], x=x[i], kr=kr, kon=kon, koff=koff)
    kr = _slice_sample(_cond_logpdf_kr, init=kr, x=x, p=p, ar=ar, br=br)
    kon = _slice_sample(_cond_logpdf_kon, init=kon, p=p, koff=koff, aon=aon, bon=bon)
    koff = _slice_sample(_cond_logpdf_koff, init=koff, p=p, kon=kon, aoff=aoff, boff=boff)
    if verbose:
      log_joint = (st.poisson(mu=kr * p).logpmf(x).sum()
                   + st.beta(a=kon, b=koff).logpdf(p).sum()
                   + Fr.logpdf(kr)
                   + Fon.logpdf(kon)
                   + Foff.logpdf(koff))
      print(f'sample {t}: {log_joint}')
  return samples
