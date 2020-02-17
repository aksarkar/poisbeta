import setuptools

setuptools.setup(
  name='poisbeta',
  description='Inference algorithms for the Poisson-Beta model',
  version='0.1',
  url='https://www.github.com/aksarkar/poisbeta',
  author='Abhishek Sarkar',
  author_email='aksarkar@uchicago.edu',
  license='MIT',
  install_requires=[
    'numpy',
    'scipy',
  ],
  packages=setuptools.find_packages('src'),
  package_dir={'': 'src'},
)
