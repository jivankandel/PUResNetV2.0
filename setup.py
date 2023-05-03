from setuptools import setup,find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
liscense=(this_directory / "LICENSE.txt").read_text()
setup(
    name='puresnet',
    version='0.0.1',
    description='PUResNetV2.0 Prediction of Protein Ligand Binding Sites',
    long_description=long_description,
    long_description_content_type='text/markdown'
    
    packages=find_packages(),
    install_requires=[
      'numpy'  
    ],
    package_data={'puresnet':['data/model90 70210.pt','data/training_data.npy','data/validation_data.npy']
                  }
)
