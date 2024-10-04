from setuptools import setup, find_packages

setup(
    name='community_connectivity',
    version='0.1',
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
    python_requires='>=3.9',  # Specify Python version compatibility
)