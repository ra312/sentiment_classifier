from setuptools import find_packages, setup
from package import Package

setup(
      version = '0.0.0',
      author="Rauan Akylzhanov",
      author_email="akylzhanov.r@gmail.com",
      packages=find_packages(),
      include_package_data=True,
      cmdclass={ "package": Package}
      )
