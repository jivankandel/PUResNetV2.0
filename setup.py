from setuptools import setup,find_packages
with open('README.md') as f:
    readme=f.read()
with open('LICENSE') as f:
    license=f.read()
setup(
    name='puresnet',
    version='0.0.1',
    description='PUResNetV2.0 Prediction of Protein Ligand Binding Sites',
    long_description=readme,
    license=license.
    packages=find_packages(),
    install_requires=[
      'numpy'  
    ],
    package_data={'puresnet':['data/model90 70210.pt','data/training_data.npy','data/validation_data.npy']
                  }
)
