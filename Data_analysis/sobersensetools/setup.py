

from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'Package to unpack and model touch data from the SoberSense app'
LONG_DESCRIPTION = 'Package to unpack and model touch data from the SoberSense app, using a neural networks through PyTorch.'


setup(
        name="sobersensetools", 
        version=VERSION,
        author="Nicholas Gregory",
        author_email="<ng432@cam.ac.uk>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
        'torch',      
        'scikit-learn',
        'numpy',     
    ],
        
        keywords=['python', 'machine learning', 'touch analysis', 'sobersense'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)