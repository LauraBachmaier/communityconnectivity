# Automatically import commonly used functions, classes, and modules
from parcels import (
    AdvectionRK4, FieldSet, JITParticle, ParticleSet, Variable, download_example_dataset,
    Field, ParcelsRandom, VectorField, DiffusionUniformKh, Geographic, GeographicPolar
)

from datetime import timedelta, datetime
from operator import attrgetter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import dask
import geopandas as gpd
from shapely.geometry import Point, Polygon
from netCDF4 import Dataset
import json
import copy
import itertools
from pygbif import occurrences
import folium
from folium import plugins
from scipy.spatial import cKDTree
import gc
import multiprocessing
from tqdm import tqdm
import os

# Initialize Dask Client for distributed processing
dask_client = dask.distributed.Client()

# Optionally, define an `__all__` to restrict what is exported from the package
__all__ = [
    'AdvectionRK4', 'FieldSet', 'JITParticle', 'ParticleSet', 'Variable', 'download_example_dataset',
    'Field', 'ParcelsRandom', 'VectorField', 'DiffusionUniformKh', 'Geographic', 'GeographicPolar',
    'timedelta', 'datetime', 'attrgetter', 'plt', 'np', 'pd', 'xr', 'dask', 'gpd', 'Point', 'Polygon',
    'Dataset', 'json', 'copy', 'itertools', 'occurrences', 'folium', 'plugins', 'cKDTree', 'gc',
    'multiprocessing', 'tqdm', 'dask_client'
]
