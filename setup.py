import os
import subprocess
from setuptools import setup, Extension, find_packages

def check_for_gcc():
    """Ensure GCC is installed before proceeding."""
    try:
        subprocess.check_call(['gcc', '--version'])
    except OSError:
        raise RuntimeError("GCC is not installed or not found in PATH. Please install GCC via conda.")

check_for_gcc()

setup(
    name='community_connectivity',
    version='0.1.0',
    description='A package for multi-species dispersal modelling using Lagrangian particle tracking in ocean models and related tasks.',
    author='Laura Bachmaier',
    author_email='laura,bachmaier@plymouth.ac.uk',
    url='https://github.com/LauraBachmaier/communityconnectivity',  # Your GitHub or project URL
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'parcels',
        'dask',
        'pandas',
        'xarray',
        'matplotlib',
        'geopandas',
        'shapely',
        'netCDF4',
        'cartopy',
        'IPython',
        'requests',
        'folium',
        'psutil',
        'copernicusmarine',
        'trajan',
        'cmocean',
        'pygbif',
        'tqdm',
    ],
    classifiers=[  # Classifiers (this helps categorize your package on PyPI or Conda)
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  # Specify Python version compatibility
)