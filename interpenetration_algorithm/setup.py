from setuptools import setup

setup(name='ISA',
      version='0.0.1',
      description='Interpenetration and scoring algortihm',
      author='AG-Peter',
      packages=['ISA'],
      install_requires=[
          'numpy',
          'mdtraj'
      ],
      zip_safe=False)
