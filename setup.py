import setuptools

setuptools.setup(
   name='SimpleRL',
   version='0.99',
   description='Make simple RL toy models',
   author='Nicolò Rossi',
   author_email='olocin.issor@gmail.com',
   install_requires=['wheel', 'torch'],
   packages=setuptools.find_packages()
)