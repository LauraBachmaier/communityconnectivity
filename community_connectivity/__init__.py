# region Setup Functions
def welcome_message():
    print("Welcome to the Community Connectivity Package!")
    print("This package facilitates multi-species dispersal models using Ocean Parcels and NEMO hydrodynamic data.")
    print("It allows you to set up, run, and analyze connectivity data for any marine species and region of your choice.")
    print("You can use the package in three different ways:")
    print("A. Interactive: Guided setup of the entire process. Recommended for first-time users.")
    print("B. Automatic: Run the entire process automatically using initial setup parameters.")
    print("C. Manual: Access individual functions to create a custom workflow.")
    print("\nPlease select an option:")

def choose_mode():
    dropdown = widgets.Dropdown(
        options=[
            ('Interactive Mode', 'community_connectivity.interactive'),
            ('Automatic Mode', 'community_connectivity.automatic'),
            ('Manual Mode', 'community_connectivity.manual'),
        ],
        description='Mode:',
    )

    button = widgets.Button(description="Submit")

    # Container for output display
    output = widgets.Output()

    display(dropdown, button, output)

    # Function to handle button click event
    def on_button_click(b):
        selected_module = dropdown.value  # Get the selected module name
        with output:
            clear_output()
            print(f"Selected Mode: {dropdown.label}")

            # Print instructions for user
            print("Interactive mode will provide a step-by-step guide to setup and analyse the connectivity model.")
            print("It uses a function called workflow which is separated into 6 steps.")
            print("To begin using this mode, please run the following commands in a new cell:")
            print(f"from {selected_module} import *")
            print("workflow_step(1)")

    # Bind the function to button click
    button.on_click(on_button_click)

def load_modules():
    """
    Function to import and initialize all required modules for Lagrangian Particle Tracking in ocean models
    and related tasks. This function assumes that the required packages have already been installed.
    """
    import importlib
    import subprocess
    import sys

    # Import necessary modules
    from parcels import (
        AdvectionRK4, FieldSet, JITParticle, ParticleSet, Variable, download_example_dataset,
        Field, ParcelsRandom, VectorField, DiffusionUniformKh, Geographic, GeographicPolar
    )

    from datetime import timedelta, datetime
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import ListedColormap
    import numpy as np
    import scipy.interpolate as interpolate
    from scipy.ndimage import binary_dilation
    import numpy.ma as ma
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
    import copy
    import itertools
    from pygbif import occurrences
    import folium
    from folium import plugins
    from scipy.spatial import cKDTree
    import psutil
    import gc
    import copernicusmarine as cm
    from tqdm import tqdm
    import ipywidgets as widgets
    import cartopy.feature as cfeature

    # Make these modules available globally
    globals().update(locals())
# endregions

# region Run Setup Function
# Import required functions
load_modules()

# Show the welcome message and allow the user to select a mode
welcome_message()
choose_mode()
#endregion

# region Create global variables
hydro_params = None
species_traits_path = None
runtime = None
traits = None
hydro_data = None
clean_spp = None
spp_params = None
landmask = None
coastalmask = None
lonarray = None
latarray = None
meshgrid = None
release_grids = None
settlement_grids = None
full_grid_df = None
grid_ids = None
fieldset = None
adjusted_runtime = None
pset = None
kernels = None
traj = None
filtered_trajectories = None
buffered_settlement_grids = None
# endregion