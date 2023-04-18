from setuptools import setup, find_packages


project_name = 'gigastep'
version = '0.0.1'
install_requires = []

setup(name=project_name,
      packages=[project_name.lower()],
      package=find_packages(include=[project_name.lower()]),
      version=version,
      install_requires=install_requires,
      include_package_data=True)