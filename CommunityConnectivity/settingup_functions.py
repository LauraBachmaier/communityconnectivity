# Setting up the package
## Load Modules Function
def load_modules():
    """
    Function to import and initialize all required modules for Lagrangian Particle Tracking in ocean models
    and related tasks. If a module is not installed, it attempts to install it using pip.
    """
    import importlib
    import subprocess
    import sys

    def install_and_import(package):
        try:
            module = importlib.import_module(package)
            globals()[package] = module
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            module = importlib.import_module(package)
            globals()[package] = module

    # List of packages to install and import
    packages = [
        'numpy',  # Numerical computing library for array operations
        'scipy',  # Scientific computing library
        'dask',  # Parallel computing library
        'dask.distributed',  # Dask distributed parallel processing (for multi-core computation)
        'pandas',  # Data manipulation and analysis library
        'xarray',  # Multi-dimensional array data structures for labeled data
        'matplotlib',  # Plotting library
        'geopandas',  # Geospatial data manipulation and analysis
        'shapely',  # Geometry objects for geospatial data manipulation
        'netCDF4',  # NetCDF file handling for array data
        'cartopy',  # Cartographic projections for geospatial plotting
        'IPython',  # IPython utilities for displaying HTML and images
        'requests',  # Used for making HTTP requests
        'folium',  # Python library for generating interactive maps
        'mpl_toolkits.basemap',  # Matplotlib toolkit for creating 2D maps
        'psutil',  # Memory availability checks
        'copernicusmarine',  # Download hydrodynamic data from Copernicus
        'trajan',  # Oceanographic and geospatial data handling (assumed to be custom or less common)
        'cmocean',  # Colormaps specifically designed for oceanographic data
        'pygbif',  # Python client for Global Biodiversity Information Facility (GBIF)
        'scipy.ndimage',  # Multi-dimensional image processing and analysis
        'glob',  # File path pattern matching
        'os',  # Operating system interface for file and directory operations
        'json',  # JSON data handling
        'copy',  # For copying objects
        'itertools',  # Iteration utilities for efficient looping
        'gc',  # Garbage collection to manage memory during simulations
        'multiprocessing',  # For handling multiprocessing
        'tqdm',  # Progress bar library
        'math',  # Standard library for mathematical functions
        'datetime',  # Standard library for manipulating dates and times
        'operator',  # Standard library for functional programming constructs
    ]

    # Loop through the packages and install/import them
    for package in packages:
        install_and_import(package)

    # Import specific functions/classes from these modules
    from parcels import (
        AdvectionRK4, FieldSet, JITParticle, ParticleSet, Variable, download_example_dataset,
        Field, ParcelsRandom, VectorField, DiffusionUniformKh, Geographic, GeographicPolar
    )

    # Import other necessary modules
    from datetime import timedelta, datetime
    from operator import attrgetter
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib import colormaps
    import numpy as np
    import scipy.interpolate as interpolate
    from scipy.ndimage import binary_dilation
    import numpy.ma as ma
    import dask.array as da
    import pandas as pd
    import trajan as ta
    import xarray as xr
    import cmocean
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    from netCDF4 import Dataset
    from glob import glob
    import os
    import cartopy.crs as ccrs
    from IPython.display import HTML, Image, display, clear_output
    import json
    import dask
    from dask.config import set as dask_set
    import copy
    import itertools
    from pygbif import occurrences
    import folium
    from folium import plugins
    from mpl_toolkits.basemap import Basemap
    from scipy.spatial import cKDTree
    import psutil
    import gc
    import copernicusmarine as cm
    from dask.distributed import Client
    import multiprocessing
    from tqdm import tqdm

    # Initialize Dask Client for distributed processing
    dask_client = Client()

    # Make these modules available globally
    globals().update(locals())