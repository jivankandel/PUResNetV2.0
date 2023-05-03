from setuptools import setup,find_packages
setup(
    name='puresnet',
    version='0.0.1',
    description='pdb to sparse',
    packages=find_packages(),
    install_requires=[
      'numpy'  
    ],
    package_data={'puresnet':['data/model90 70210.pt','data/training_data.npy','data/validation_data.npy']
                  }
)