
# region Interactive Functions

# region Import Modules
def load_modules():
    """
    Function to import and initialize all required modules for Lagrangian Particle Tracking in ocean models
    and related tasks. If a module is not installed, it attempts to install it using pip.
    """
    import importlib
    import subprocess
    import sys

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
    import math

    # Make these modules available globally
    globals().update(locals())

# endregion

# region Processing the Species Trait Database
# Function to calculate dispersal period for each species and determine the overall time range
def calculate_overall_dispersal_period(traits):
    """
    Calculates the overall dispersal period (earliest spawning start to latest dispersal end)
    based on all species' data.

    Args:
        traits (list): List of dictionaries with species traits.

    Returns:
        tuple: Overall start and end dates for dispersal period (as strings).
    """
    overall_start = None
    overall_end = None

    # Loop through all species
    for species in traits:
        # Parse spawning dates
        spawning_start_date = datetime.strptime(species['spawning_start'], "%d/%m/%Y")
        spawning_end_date = datetime.strptime(species['spawning_end'], "%d/%m/%Y")
        max_pld = int(species['max_PLD'])

        # Calculate the dispersal end date by adding max PLD to the spawning end date
        dispersal_end_date = spawning_end_date + timedelta(days=max_pld)

        # Update overall start and end dates
        if overall_start is None or spawning_start_date < overall_start:
            overall_start = spawning_start_date
        if overall_end is None or dispersal_end_date > overall_end:
            overall_end = dispersal_end_date

    return overall_start.strftime('%Y-%m-%d'), overall_end.strftime('%Y-%m-%d')

# Function to load species traits from the CSV file
# Function to load species traits from the CSV file
def load_species_traits_from_csv():
    """
    Prompts the user to provide a CSV file with species traits and processes the data.

    Returns:
        dict: Dictionary containing the species traits, including overall dispersal period.
    """
    species_traits_path = input("Enter the input path and file name for your species traits data: ").strip()
    if not species_traits_path.endswith(".csv"):
        species_traits_path += ".csv"

    if not os.path.exists(species_traits_path):
        print(f"File not found: {species_traits_path}")
        return None

    traits = []

    # Open and read the CSV file
    with open(species_traits_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            traits.append({
                "species_name": row['species_name'],
                "spawning_start": row['spawning_start'],
                "spawning_end": row['spawning_end'],
                "min_PLD": int(row['min_PLD']),
                "max_PLD": int(row['max_PLD']),
                "average_PLD": int(row['average_PLD']),
                "fecundity": int(row['fecundity'])
            })

    # Calculate the overall dispersal period for all species
    overall_start, overall_end = calculate_overall_dispersal_period(traits)

    print(f"Overall dispersal period (all species): {overall_start} to {overall_end}")

    # Return the overall dispersal period along with the species data
    return {
        "overall_dispersal_start": overall_start,
        "overall_dispersal_end": overall_end
    }, species_traits_path, traits

# endregion

# region Setting the Hydrodynamic Parameters
## Collecting Hydrodynamic Parameters
# Collecting Hydrodynamic Parameters
# Function to collect hydrodynamic parameters and calculate time range and runtime
def collect_hydro_params():
    """
    Collects the hydrodynamic data parameters from the user and incorporates species traits for time range calculation.

    Returns:
        dict: Dictionary containing the user-provided parameters.
        str: Path to the species traits CSV file.
        timedelta: The total runtime (dispersal duration) in days.
    """
    username = input("Enter your username: ")
    password = input("Enter your password: ")
    dataset_id = input("Enter the dataset ID: ")
    longitude_range = [
        float(input("Enter the minimum longitude: ")),
        float(input("Enter the maximum longitude: "))
    ]
    latitude_range = [
        float(input("Enter the minimum latitude: ")),
        float(input("Enter the maximum latitude: "))
    ]

    # Load species traits from the CSV file and calculate the overall dispersal period
    dispersal_time, species_traits_path, traits  = load_species_traits_from_csv()
    if not dispersal_time:
        print("Error loading species traits. Aborting operation.")
        return None, None, None

    # Use the overall dispersal period from the species traits as the default time range
    time_range = [
        datetime.strptime(dispersal_time["overall_dispersal_start"], '%Y-%m-%d'),
        datetime.strptime(dispersal_time["overall_dispersal_end"], '%Y-%m-%d')
    ]

    # Calculate the runtime in days from the start to the end of the hydrodynamic data
    runtime = (time_range[1] - time_range[0]).days  # timedelta object representing the total time in days

    print(f"Runtime (in days): {runtime} days")

    # Prompt the user for export option
    export_input = input("Do you want to export the data to a NetCDF file? (yes/no): ").strip().lower()
    export = True if export_input in ["yes", "y"] else False

    # Get the output path from the user without file extension
    output_path = input("Enter the output path and file name: ")

    # Ensure that the file has a .nc extension
    if not output_path.endswith(".nc"):
        output_path += ".nc"

    return {
        "dataset_id": dataset_id,
        "longitude_range": longitude_range,
        "latitude_range": latitude_range,
        "time_range": time_range,
        "username": username,
        "password": password,
        "output_path": output_path,
        "export": export,
        "traits": traits,
        "species_traits_path": species_traits_path,
        "runtime": runtime
    }

## Edit Hydrodynamic Parameters
def edit_hydro_params(hydro_params):
    """
    Allows the user to edit existing parameters.

    Args:
        hydro_params (dict): Dictionary of existing parameters.

    Returns:
        dict: Updated dictionary of parameters.
    """
    # Define a list of keys that are lists and their expected types
    list_keys = {
        'longitude_range': float,
        'latitude_range': float,
        'time_range': str  # Time range will be treated as datetime objects
    }

    # Iterate over each parameter in the dictionary
    for key, value in hydro_params.items():
        edit = input(f"Do you want to change {key} (current value: {value})? (yes/no): ").strip().lower()

        if edit in ["yes", "y"]:
            if key in list_keys:
                # Handle list values separately (longitude_range, latitude_range, time_range)
                if len(value) == 2:
                    if key == 'time_range':
                        # Handle date input for time_range
                        try:
                            new_value = [
                                datetime.strptime(input(
                                    f"Enter the new start date for {key} (YYYY-MM-DD) (current value: {value[0].strftime('%Y-%m-%d')}): "),
                                    '%Y-%m-%d'),
                                datetime.strptime(input(
                                    f"Enter the new end date for {key} (YYYY-MM-DD) (current value: {value[1].strftime('%Y-%m-%d')}): "),
                                    '%Y-%m-%d')
                            ]
                        except ValueError:
                            print("Invalid date format. Please enter dates in YYYY-MM-DD format.")
                            continue
                    else:
                        # Handle numeric values for latitude_range, longitude_range
                        try:
                            new_value = [
                                list_keys[key](input(
                                    f"Enter the new value for {key} - first element (current value: {value[0]}): ")),
                                list_keys[key](input(
                                    f"Enter the new value for {key} - second element (current value: {value[1]}): "))
                            ]
                        except ValueError:
                            print(f"Invalid input. Please enter values of type {list_keys[key].__name__}.")
                            continue
                else:
                    print(f"Expected a list with 2 elements for {key}.")
                    continue
            else:
                # For non-list values, just update the parameter
                new_value = input(f"Enter the new value for {key} (current value: {value}): ")
                # Try to cast numeric values (if possible) to their original types
                try:
                    if isinstance(value, bool):
                        new_value = new_value.lower() in ['true', 'yes', '1']
                    elif isinstance(value, (int, float)):
                        new_value = type(value)(new_value)
                except ValueError:
                    print(f"Invalid input. Keeping the original value for {key}.")
                    continue

            # Update the parameter in the dictionary
            hydro_params[key] = new_value

    # Convert the time_range back to strings before saving
    hydro_params['time_range'] = [
        hydro_params['time_range'][0].strftime('%Y-%m-%d'),
        hydro_params['time_range'][1].strftime('%Y-%m-%d')
    ]

    # Save the updated parameters to the JSON file
    with open("hydrodynamic_parameters.json", "w") as file:
        json.dump(hydro_params, file, indent=4)

    return hydro_params

## Load Hydrodynamic Parameters
def params():
    """
    Prompts the user to input hydrodynamic data parameters, allows for reusing or editing existing parameters,
    and then saves them to a JSON file.

    Returns:
        dict: Dictionary containing all the user-provided parameters (excluding species traits, path, and runtime).
        str: The species traits path.
        timedelta: The runtime in days.
    """
    params_file = "hydrodynamic_parameters.json"

    # Check if parameters file exists and offer to reuse
    if os.path.exists(params_file):
        reuse_input = input("Found existing parameters. Do you want to reuse them? (yes/no): ").strip().lower()
        if reuse_input in ["yes", "y"]:
            with open(params_file, "r") as file:
                hydro_params = json.load(file)

                # Convert runtime from days back to a timedelta object
                #runtime = timedelta(days=hydro_params['runtime'])

                # Convert time_range from strings back to datetime objects
                hydro_params['time_range'] = [
                    datetime.strptime(hydro_params['time_range'][0], '%Y-%m-%d'),
                    datetime.strptime(hydro_params['time_range'][1], '%Y-%m-%d')
                ]

                print("Reusing existing parameters:")
                for key, value in hydro_params.items():
                    print(f"{key}: {value}")

            # Offer to edit specific parameters
            edit_input = input("Do you want to edit any parameters? (yes/no): ").strip().lower()
            if edit_input in ["yes", "y"]:
                hydro_params = edit_hydro_params(hydro_params)

            # Save runtime as days in the JSON file
            #hydro_params['runtime'] = runtime.days

            # Convert the time_range back to strings before saving
            hydro_params['time_range'] = [
                hydro_params['time_range'][0].strftime('%Y-%m-%d'),
                hydro_params['time_range'][1].strftime('%Y-%m-%d')
            ]

            # Save the (possibly edited) parameters back to the file
            with open(params_file, "w") as file:
                json.dump(hydro_params, file)

            # Extract species traits, path, and return runtime as timedelta
            species_traits_path = hydro_params.get("species_traits_path", None)
            traits = hydro_params.get("traits", None)
            runtime = hydro_params.get("runtime", None)

            # Remove the traits and species traits path before returning hydro_params
            hydro_params.pop("traits", None)
            hydro_params.pop("species_traits_path", None)
            hydro_params.pop("runtime", None)

            return hydro_params, species_traits_path, runtime, traits

    # If not reusing or no existing file, prompt the user for input
    hydro_params = collect_hydro_params()

    # Offer to review and edit before saving
    review_input = input("Do you want to review and edit the parameters before saving? (yes/no): ").strip().lower()
    if review_input in ["yes", "y"]:
        hydro_params = edit_hydro_params(hydro_params)

    # Calculate runtime as timedelta
    runtime = hydro_params['runtime']

    # Save runtime as days in the JSON file
    hydro_params['runtime'] = runtime.days

    # Convert the time_range back to strings before saving
    hydro_params['time_range'] = [
        hydro_params['time_range'][0].strftime('%Y-%m-%d'),
        hydro_params['time_range'][1].strftime('%Y-%m-%d')
    ]

    # Save the parameters to a JSON file
    with open(params_file, "w") as file:
        json.dump(hydro_params, file)

    print("Parameters have been saved to 'hydrodynamic_parameters.json'.")

    # Extract species traits and species traits path
    traits = hydro_params.get("traits", None)
    species_traits_path = hydro_params.get("species_traits_path", None)

    # Remove the species traits, path, and runtime from hydro_params before returning
    hydro_params.pop("traits", None)
    hydro_params.pop("species_traits_path", None)
    hydro_params.pop("runtime", None)

    return hydro_params, species_traits_path, runtime, traits
# endregion

# region Downloading Hydrodynamic Data
def download_hydrodynamic_data(**kwargs):
    """
    Downloads hydrodynamic data from the Copernicus Marine Service.

    Args:
        username (str): Username for Copernicus Marine Service.
        password (str): Password for Copernicus Marine Service.
        dataset_id (str): Dataset ID for the hydrodynamic data.
        longitude_range (list): Range of longitudes [min_longitude, max_longitude].
        latitude_range (list): Range of latitudes [min_latitude, max_latitude].
        time_range (list): Range of time ["start_date", "end_date"].
        output_path (str): Path to save the downloaded data.
        compression_level (int, optional): Compression level for the output NetCDF file. Defaults to 9.
    """

    # Extract parameters from kwargs
    username = kwargs.get("username")
    password = kwargs.get("password")
    dataset_id = kwargs.get("dataset_id")
    longitude_range = kwargs.get("longitude_range")
    latitude_range = kwargs.get("latitude_range")
    time_range = kwargs.get("time_range")
    output_path = kwargs.get("output_path")
    export = kwargs.get("export", True)

    # Set parameters
    data_request = {
        "dataset_id": dataset_id,
        "longitude": longitude_range,
        "latitude": latitude_range,
        "time": time_range
    }

    # Load xarray dataset
    data = cm.open_dataset(
        dataset_id=data_request["dataset_id"],
        minimum_longitude=data_request["longitude"][0],
        maximum_longitude=data_request["longitude"][1],
        minimum_latitude=data_request["latitude"][0],
        maximum_latitude=data_request["latitude"][1],
        start_datetime=data_request["time"][0],
        end_datetime=data_request["time"][1],
        username=username,
        password=password
    )

    # Save to Netcdf
    if export:
        # Check if output_path is provided
        if not output_path:
            raise ValueError("output_path must be provided if export is True.")

        # Define compression options
        encoding = {var: {'zlib': True, 'complevel': 9} for var in data.data_vars}

        # Export the compressed file
        data.to_netcdf(output_path, encoding=encoding)

        # Load the data back into memory
        hydro_data = xr.open_dataset(output_path)
    else:
        # Directly use the data without saving to a file
        hydro_data = data

    return hydro_data
# endregion

# region Downloading Species Occurrence Data
## region Set Up Functions
### Edit Species Parameters
def edit_spp_params(spp_params):
    """
    Allows the user to edit existing parameters in one streamlined interaction.

    Args:
        spp_params (dict): Dictionary of existing parameters.

    Returns:
        dict: Updated dictionary of parameters.
    """
    # Iterate over each parameter in the dictionary
    for key, value in spp_params.items():
        edit = input(f"Do you want to change {key} (current value: {value})? (yes/no): ").strip().lower()
        if edit in ["yes", "y"]:
            # Collect new input for that parameter
            if isinstance(value, list) and len(value) == 2:
                if key == 'geojson_file_path':
                    new_value = input(f"Enter the new value for {key} (current value: {value}): ")
                else:
                    # For list-type values like longitude/latitude ranges
                    new_value = [
                        input(f"Enter the new first value for {key} (current value: {value[0]}): "),
                        input(f"Enter the new second value for {key} (current value: {value[1]}): ")
                    ]
            else:
                new_value = input(f"Enter the new value for {key} (current value: {value}): ")

            # Update the parameter in the dictionary
            spp_params[key] = new_value

    # Save the updated parameters
    return spp_params

### Collect Species Occurrence Parameters
def collect_spp_occ_params(species_traits_path):
    """
    Collects the species occurrence parameters from the user in one streamlined prompt.

    Returns:
        dict: Dictionary containing the user-provided parameters.
    """

    # Check if the user wants to reuse an existing GeoJSON file
    print("This package requires a GeoJSON file of your study area.")
    print("You can either upload your own file or use the package functionality to select your study area.")
    geojson_reuse = input("Do you want to use an existing GeoJSON file? (yes/no): ").strip().lower()

    if geojson_reuse in ["yes", "y"]:
        # Prompt for GeoJSON file path, adding the .geojson extension if not provided
        geojson_file_path = input("Enter the input path and file name: ").strip()
        if not geojson_file_path.endswith(".geojson"):
            geojson_file_path += ".geojson"
    else:
        geojson_file_path = None

    # Check if the user wants to save the results as a CSV file
    save_csv = input("Do you want to save the species occurrence data as a CSV file? (yes/no): ").strip().lower() == 'yes'
    if save_csv:
        # Prompt for CSV file save path, adding the .csv extension if not provided
        csv_file_path = input("Enter the output path and file name: ").strip()
        if not csv_file_path.endswith(".csv"):
            csv_file_path += ".csv"
    else:
        csv_file_path = None

    return {
        'species_traits_path': species_traits_path,
        'geojson_file_path': geojson_file_path,
        'save_csv': save_csv,
        'csv_file_path': csv_file_path
    }

### Create Map
def create_map():
    """
    Creates an interactive map for the user to draw a polygon and save it as a GeoJSON file.

    Ensures that the user provides a valid file path after saving the GeoJSON file.

    Returns:
        str: The valid file path of the saved GeoJSON file.
    """
    # Step 1: Create the interactive map with drawing capabilities
    m = folium.Map()
    draw = plugins.Draw(
        export=True,
        draw_options={'polygon': True, 'polyline': False, 'rectangle': False, 'circle': False, 'marker': False,
                      'circlemarker': False}
    )
    draw.add_to(m)
    display(m)

    # Step 2: Provide instructions to the user
    print("Please follow these steps to define your study area:")
    print("1. Use the drawing tool on the map to draw a polygon around your area of interest.")
    print("2. After drawing, click the 'Export' button on the map to save the polygon as a GeoJSON file.")
    print("3. Once you have saved the file, return here and enter the full path to your saved GeoJSON file.")

    # Step 3: Prompt user for the file path and ensure the file exists
    file_path = input("Enter the input path and file name: ")
    # Ensure that the file has a .nc extension
    if not file_path.endswith(".geojson"):
        file_path += ".geojson"
    while not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found. Please make sure it was saved correctly.")
        file_path = input("Enter the input path and file name: ")
        if not file_path.endswith(".geojson"):
            file_path += ".geojson"

    # Step 4: Return the valid file path
    return file_path

### Read GeoJSON File
def read_geojson(geojson_file_path):
    """
    Reads a GeoJSON file and extracts the polygon coordinates in both WKT format and as a Shapely Polygon object.

    Args:
        file_path (str): The path to the GeoJSON file.

    Returns:
        tuple: A tuple containing:
            - WKT format string of the polygon.
            - Shapely Polygon object.
    """
    with open(geojson_file_path, 'r') as geojson_file:
        geojson_data = json.load(geojson_file)

    # Extract polygon coordinates
    polygon_coords = geojson_data['features'][0]['geometry']['coordinates'][0]

    # Ensure polygon is closed
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords.append(polygon_coords[0])

    # Create WKT format string
    geometry = 'POLYGON((' + ', '.join([f'{lon} {lat}' for lon, lat in polygon_coords]) + '))'

    # Create Shapely Polygon object
    shapely_polygon = Polygon(polygon_coords)

    return geometry, shapely_polygon

### Get Species Occurrences
def get_spp_occ(species_name, geometry, limit=300):
    """
    Fetches species occurrences from GBIF within the given polygon.

    Args:
        species_name (str): The scientific name of the species.
        geometry (str): The polygon geometry in WKT format.
        limit (int): The number of records to fetch per request.

    Returns:
        list: A list of species occurrence records.
    """
    all_records = []
    offset = 0

    while True:
        results = occurrences.search(scientificName=species_name, geometry=geometry, limit=limit, offset=offset)
        for record in results.get('results', []):
            if 'decimalLatitude' in record and 'decimalLongitude' in record:
                all_records.append({
                    'species': species_name,
                    'latitude': record['decimalLatitude'],
                    'longitude': record['decimalLongitude']
                })
        offset += limit
        if len(results.get('results', [])) < limit:
            break

    return all_records

### Landmask and Coastalmask
def create_masks(hydro_data):
    """
    Creates and plots landmask and coastal mask.

    Args:
        hydrodata (Dataset): The dataset containing the hydro data with land and ocean information.

    Returns:
        tuple: landmask (numpy.ndarray), coastalmask (numpy.ndarray), lon_centers (numpy.ndarray), lat_centers (numpy.ndarray)
    """

    def make_coastalmask(landmask):
        """Creates a coastal mask from the landmask by identifying adjacent ocean cells."""
        coastal_mask = binary_dilation(landmask, structure=np.ones((3, 3))) & ~landmask
        return coastal_mask.astype(int)

    # Check the number of dimensions in the 'uo' variable
    var = hydro_data.variables["uo"]

    # Determine the number of dimensions
    num_dims = len(var.shape)

    if num_dims == 3:
        # 2D case (e.g., (lat, lon, time))
        landmask = var[0, :, :]

    elif num_dims == 4:
        # 3D case (e.g., (time, lat, lon, depth))
        # Here you can choose which slice to use. For example, using the first time step:
        landmask = var[0, 0, :, :]

    else:
        raise ValueError("Unsupported data dimensionality: {}".format(num_dims))

    # Mask invalid data in the 'uo' variable
    landmask = np.ma.masked_invalid(landmask)

    # Convert the mask to an integer array: 1 for valid land, 0 for invalid (ocean)
    landmask = landmask.mask.astype("int")

    # Generate the coastal mask
    coastalmask = make_coastalmask(landmask)

    # Extract latitude and longitude data and create meshgrid
    latitudes = hydro_data.variables['latitude'][:]
    longitudes = hydro_data.variables['longitude'][:]

    # Create 2D meshgrid from latitudes and longitudes
    lonarray, latarray = np.meshgrid(longitudes, latitudes)

    # Plotting
    lons_plot = lonarray
    lats_plot = latarray

    # Define the grid resolution
    dlon = 1 / 12
    dlat = 1 / 12

    # Calculate the centers of the grid cells
    x = hydro_data.variables["longitude"][:-1] + np.diff(hydro_data.variables["longitude"]) / 2
    y = hydro_data.variables["latitude"][:-1] + np.diff(hydro_data.variables["latitude"]) / 2
    lon_centers, lat_centers = np.meshgrid(x, y)

    # Define the colors for the masks
    color_land = '#008000'  # Green
    color_coastal = '#A52A2A'  # Brown/Red
    color_ocean = '#0000FF'  # Blue

    # Create color maps for land and coastal masks
    cmap_land = ListedColormap([color_land, 'none'])
    cmap_coastal = ListedColormap([color_coastal, 'none'])

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot Landmask
    landmask_plot = np.where(landmask, 1, np.nan)  # Land in green
    coastalmask_plot = np.where(coastalmask, 1, np.nan)  # Coastal in brown/red
    ocean_points = np.logical_and(landmask == 0, coastalmask == 0)

    ax.pcolormesh(
        lons_plot,
        lats_plot,
        landmask_plot,
        cmap=cmap_land,
        shading="auto",
        alpha=0.5
    )

    ax.pcolormesh(
        lons_plot,
        lats_plot,
        coastalmask_plot,
        cmap=cmap_coastal,
        shading="auto",
        alpha=0.5
    )

    ax.scatter(
        lons_plot[ocean_points],
        lats_plot[ocean_points],
        c=color_ocean,
        s=1,
        label="Ocean Point",
    )

    ax.scatter(
        lons_plot[coastalmask == 1],
        lats_plot[coastalmask == 1],
        c=color_coastal,
        s=1,
        label="Coastal Point",
    )

    ax.set_title("Land and Coastal Mask", fontsize=11)
    ax.set_ylabel("Latitude [degrees]")
    ax.set_xlabel("Longitude [degrees]")

    custom_lines = [
        Line2D([0], [0], color=color_land, marker="o", markersize=10, markeredgecolor="k", lw=0),
        Line2D([0], [0], color=color_coastal, marker="o", markersize=10, markeredgecolor="k", lw=0),
        Line2D([0], [0], color=color_ocean, marker="o", markersize=10, markeredgecolor="k", lw=0),
    ]
    ax.legend(
        custom_lines,
        ["Land Point", "Coastal Point", "Ocean Point"],
        bbox_to_anchor=(0.01, 0.93),
        loc="center left",
        borderaxespad=0.0,
        framealpha=1,
    )

    plt.tight_layout()
    plt.show()

    return landmask, coastalmask, lonarray, latarray

### Clean Species Data
def clean_spp_occ(distributions, landmask, coastalmask, lonarray, latarray):
    """
    Cleans species data by filtering for coastal species locations.
    Interactively asks the user if they want to save the cleaned data.

    Parameters:
    - distributions: DataFrame containing species occurrence data with 'latitude' and 'longitude' columns.
    - landmask: 2D array indicating land (1) and ocean (0).
    - coastal_mask: 2D array indicating coastal regions.
    - lat_centers: Array of latitude grid points.
    - lon_centers: Array of longitude grid points.

    Returns:
    - coastal_species_data: DataFrame of species found in coastal regions.
    """
    # Extract latitude and longitude from the distributions dataset
    species_lats = distributions['latitude'].values
    species_lons = distributions['longitude'].values

    # Convert species coordinates to grid indices
    lat_indices = np.digitize(species_lats, latarray[:, 0])
    lon_indices = np.digitize(species_lons, lonarray[0, :])

    # Check if these indices fall into coastal cells
    coastal_cells = np.zeros_like(landmask)
    for lat_idx, lon_idx in zip(lat_indices, lon_indices):
        if lat_idx < landmask.shape[0] and lon_idx < landmask.shape[1]:
            if landmask[lat_idx, lon_idx] == 0:  # Ocean
                if coastalmask[lat_idx, lon_idx] == 1:  # Coastal
                    coastal_cells[lat_idx, lon_idx] = 1

    # Filter species data based on coastal cells
    is_coastal = np.array([
        coastal_cells[lat_idx, lon_idx] == 1
        for lat_idx, lon_idx in zip(lat_indices, lon_indices)
    ])

    clean_spp = distributions[is_coastal]

    return clean_spp

### Create Species Distribution Map
def spp_distribution_map(clean_spp):
    """
    Plots the species distributions on a map using Cartopy.

    Args:
        distributions (DataFrame): A pandas DataFrame containing species distribution data.
    """
    # Set latitude and longitude bounds with a small buffer
    lat_min = clean_spp['latitude'].min() - 0.1
    lat_max = clean_spp['latitude'].max() + 0.1
    lon_min = clean_spp['longitude'].min() - 0.1
    lon_max = clean_spp['longitude'].max() + 0.1

    # Set up the plot with Cartopy's Mercator projection
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.Mercator()})

    # Set map extent (bounding box) with the latitude/longitude range
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Add map features: coastlines, countries, and land/ocean
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgreen')
    ax.add_feature(cfeature.OCEAN, facecolor='lightskyblue')

    # Set up colors for the species
    colors = colormaps.get_cmap('tab10')
    handles = []

    # Plot species distributions
    for i, (species, group) in enumerate(clean_spp.groupby('species')):
        # Convert latitude and longitude to the map projection
        sc = ax.scatter(
            group['longitude'].values, group['latitude'].values,
            color=colors(i % len(colors.colors)), label=species,
            marker='o', s=50, transform=ccrs.PlateCarree()
        )
        handles.append(sc)

    # Add a title and a legend
    plt.title('Species Distributions')
    plt.legend(handles, [h.get_label() for h in handles], loc='lower right', borderaxespad=0.0, ncol=3,
               fontsize='small', title='Species')
    plt.tight_layout()

    # Display the plot
    plt.show()

# endregion

## region Complete Species Occurrence Data Download
def spp_occ(hydro_data, species_traits_path):
    """
    This function handles species occurrence data extraction, allowing for reusing or editing
    previously saved parameters or drawing a new polygon to define the study area.
    """
    params_file = "spp_occ_params.json"

    # Step 1: Check if a previous set of parameters exists and reuse or edit them
    if os.path.exists(params_file):
        reuse_input = input("Found existing parameters. Do you want to reuse them? (yes/no): ").strip().lower()
        if reuse_input in ["yes", "y"]:
            with open(params_file, 'r') as file:
                spp_params = json.load(file)
                print("Reusing existing parameters:")
                for key, value in spp_params.items():
                    print(f"{key}: {value}")

            # Offer to edit specific parameters
            edit_input = input("Do you want to edit any parameters? (yes/no): ").strip().lower()
            if edit_input in ["yes", "y"]:
                spp_params = edit_spp_params(spp_params)
        else:
            # Collect new parameters if not reusing
            spp_params = collect_spp_occ_params(species_traits_path)
    else:
        # If no parameters file exists, collect new parameters
        spp_params = collect_spp_occ_params(species_traits_path)

    # Save updated parameters after any potential edits
    with open(params_file, 'w') as file:
        json.dump(spp_params, file, indent=4)

    # Step 2: Use the geojson file path if it exists
    geojson_file_path = spp_params.get('geojson_file_path')

    # Strict check: Only create a new map if the geojson file path is missing, None, or invalid
    if not geojson_file_path or not os.path.exists(geojson_file_path):
        geojson_file_path = create_map()  # Trigger map creation if needed
        spp_params['geojson_file_path'] = geojson_file_path

        # Save the new geojson path back to the parameters file
        with open(params_file, 'w') as file:
            json.dump(spp_params, file, indent=4)
    else:
        print(f"Using pre-existing GeoJSON file: {geojson_file_path}")

    # Step 3: Read the GeoJSON file
    geometry = read_geojson(geojson_file_path)

    # Step 4: Load the species traits CSV
    traits = pd.read_csv(spp_params['species_traits_path'])

    # Step 5: Download species occurrences for each species
    all_data = []
    species_list = traits['species_name'].tolist()
    for species_name in species_list:
        all_data.extend(get_spp_occ(species_name, geometry))

    distributions = pd.DataFrame(all_data)

    # Step 6: Create landmask and clean up species data
    landmask, coastalmask, lonarray, latarray = create_masks(hydro_data)
    clean_spp = clean_spp_occ(distributions, landmask, coastalmask, lonarray, latarray)

    # Step 7: Save to CSV if requested
    if spp_params.get('save_csv') and spp_params.get('csv_file_path'):
        distributions.to_csv(spp_params['csv_file_path'], index=False)
        print(f"Data saved to {spp_params['csv_file_path']}")

    # Step 8: Create a map of the species distributions
    spp_distribution_map(clean_spp)

    return clean_spp, spp_params, landmask, coastalmask, lonarray, latarray
# endregion

# endregion

# region Setting up the Grids
## region Setup Functions
def create_grid(coastalmask, lonarray, latarray):
    """
    Creates all necessary grids based on input hydrodynamic data.

    Returns:
    - latarray: 2D array of latitude values.
    - lonarray: 2D array of longitude values.
    - full_grid: 2D array of grid points.
    - full_grid_df: DataFrame of grid points with lat/lon and grid_id.
    - coastal_grid: 2D array of flattened grid points in [longitude, latitude] format for coastal regions.
    - grid_ids: 2D array of grid IDs corresponding to each grid cell.
    """

    # Step 1: Flatten the grid to create full_grid_points
    full_grid = np.column_stack((lonarray.ravel(), latarray.ravel()))

    # Generate grid IDs
    grid_ids = np.arange(full_grid.shape[0]).reshape(lonarray.shape)

    # Create a DataFrame with full grid data
    full_grid_df = pd.DataFrame({
        'longitude': full_grid[:, 0],
        'latitude': full_grid[:, 1],
        'grid_id': grid_ids.ravel()
    })

    # Step 2: Filter coastal points
    coastal_cells = coastalmask.ravel()
    coastal_grid = full_grid[coastal_cells == 1]

    return full_grid, full_grid_df, coastal_grid, grid_ids

## Map Species to Grid
def map_spp_to_grid(full_grid_df, clean_spp):
    """
    Maps species occurrence data to the nearest centroid in the grid and assigns grid IDs.

    Parameters:
    - grid_df: DataFrame with columns 'longitude', 'latitude', and 'grid_id'.
    - species_data: DataFrame with columns 'latitude', 'longitude', and 'species'.

    Returns:
    - all_species_grid: Numpy array of unique latitudes and longitudes of all species occurrences, ensuring unique grid_ids.
    - species_grids: Dictionary of DataFrames, each containing species-specific data (longitude, latitude, grid_id) with unique lat/lon pairs per species.
    """
    # Extract grid coordinates and IDs
    grid_coords = full_grid_df[['longitude', 'latitude']].values
    grid_ids = full_grid_df['grid_id'].values

    # Create KDTree for the grid points
    grid_tree = cKDTree(grid_coords)

    # Initialize a dictionary to store each species-specific DataFrame
    species_grids = {}

    # List to collect all species grid points (for the all_species_grid output)
    all_species_points = []
    all_species_grid_ids = []

    # Process each species separately
    for species in clean_spp['species'].unique():
        # Subset the species_data for the current species
        species_subset = clean_spp[clean_spp['species'] == species]
        species_coords = np.array(list(zip(species_subset['longitude'], species_subset['latitude'])))

        # Find the nearest grid points for this species occurrence
        _, indices = grid_tree.query(species_coords)

        # Retrieve the corresponding grid points and their IDs
        nearest_grid_points = grid_coords[indices]
        nearest_grid_ids = grid_ids[indices]

        # Create a DataFrame for this species, ensuring unique (longitude, latitude) pairs
        species_df = pd.DataFrame({
            'species': species,
            'longitude': nearest_grid_points[:, 0],
            'latitude': nearest_grid_points[:, 1],
            'grid_id': nearest_grid_ids
        }).drop_duplicates(subset=['longitude', 'latitude'])  # Ensure unique lat/lon pairs

        # Store the species-specific DataFrame in the dictionary
        species_grids[species] = species_df

        # Append species grid points and grid IDs to the lists for all_species_grid
        all_species_points.append(species_df[['longitude', 'latitude']].values)
        all_species_grid_ids.append(species_df['grid_id'].values)

    # Concatenate all species grid points into a single numpy array for all_species_grid
    all_species_points = np.vstack(all_species_points)
    all_species_grid_ids = np.hstack(all_species_grid_ids)

    # Combine the grid points and IDs, and drop duplicates based on grid IDs (ensures unique grid_id)
    unique_all_species = pd.DataFrame({
        'longitude': all_species_points[:, 0],
        'latitude': all_species_points[:, 1],
        'grid_id': all_species_grid_ids
    }).drop_duplicates(subset=['grid_id'])

    # Extract the unique grid points for all_species_grid
    all_species_grid = unique_all_species[['longitude', 'latitude']].values

    return all_species_grid, species_grids

## Filter Coastal Grid
def filter_by_polygon(coastal_grid, shapely_polygon):
    """
    Filters the coastal grid to include only points within the polygon.
    The polygon is created using the read read_geojon function and crops the coastal_grid for the extent selected by the user when downloading species occurence data.

    Args:
        coastal_grid (numpy.ndarray): The coastal grid of longitude, latitude pairs.
        polygon (Polygon): The polygon to filter by (from shapely.geometry).

    Returns:
        numpy.ndarray: The filtered coastal grid.
    """
    # Create a Shapely Point for each grid cell and filter by the polygon
    filtered_grid = np.array([
        [lon, lat] for lon, lat in coastal_grid
        if shapely_polygon.contains(Point(lon, lat))
    ])
    return filtered_grid

## Save Grids
def save_grids(release_grids=None, settlement_grids=None, full_grid=None):
    """
    Optionally saves release grids, settlement grids, and meshgrid data to CSV files.

    Parameters:
    - release_grids: Dictionary containing release grid points for species.
    - settlement_grids: Dictionary containing settlement grid points with grid IDs.
    - full_grid: DataFrame or array of meshgrid points (lon, lat, grid_id).
    """

    # Optionally save the full meshgrid
    if full_grid is not None:
        save_csv = input("Do you want to save the meshgrid data to a CSV file? (yes/no): ").strip().lower()
        if save_csv in ['yes', 'y']:
            file_path = input("Enter the output path and file name: ").strip()
            # Ensure that the file has a .nc extension
            if not file_path.endswith(".csv"):
                file_path += ".csv"
            if isinstance(full_grid, pd.DataFrame):
                full_grid.to_csv(file_path, index=False)
            else:
                # If it's an array, convert it to a DataFrame for saving
                full_grid_df = pd.DataFrame(full_grid,
                                            columns=['longitude', 'latitude', 'grid_id'][:full_grid.shape[1]])
                full_grid_df.to_csv(file_path, index=False)
            print(f"Meshgrid data saved as {file_path}.")

    # Save release grids if available
    if release_grids and len(release_grids) > 0:  # Check if the list is not empty
        save_csv = input("Do you want to save the release grid points to CSV files? (yes/no): ").strip().lower()
        if save_csv in ['yes', 'y']:
            file_path = input("Enter the output path and file name: ").strip()
            if not os.path.exists(file_path):
                os.makedirs(file_path)  # Create the directory if it doesn't exist
            for species_name, df in release_grids.items():
                if isinstance(df, pd.DataFrame):
                    file_name = os.path.join(file_path, f"{species_name}_release_grid.csv")
                    df.to_csv(file_name, index=False)
                    print(f"Release grid for species '{species_name}' saved as {file_name}.")
                else:
                    print(f"Invalid data format for species '{species_name}', skipping.")

    # Save settlement grids if available
    if settlement_grids and len(settlement_grids) > 0:  # Check if the list is not empty
        save_csv = input("Do you want to save the settlement grid points to CSV files? (yes/no): ").strip().lower()
        if save_csv in ['yes', 'y']:
            file_path = input("Enter output path and file name: ").strip()
            if not os.path.exists(file_path):
                os.makedirs(file_path)  # Create the directory if it doesn't exist
            for grid_id, df in settlement_grids.items():
                if isinstance(df, pd.DataFrame):
                    file_name = os.path.join(file_path, f"settlement_grid_{grid_id}.csv")
                    df.to_csv(file_name, index=False)
                    print(f"Settlement grid for grid ID '{grid_id}' saved as {file_name}.")
                else:
                    print(f"Invalid data format for grid ID '{grid_id}', skipping.")

# endregion

## region Setup Particle Release and Settlement Strategy
def setup_release_settlement(coastalmask, clean_spp, spp_params, lonarray, latarray):
    """
    Sets up the release and settlement grids based on user choices.

    Parameters:
    - hydrodata: NetCDF dataset containing latitude, longitude, and other relevant hydrodynamic data.
    - species_data: DataFrame with columns 'latitude', 'longitude', and 'species'.
    - geojson_file: Path to GeoJSON file for filtering coastal grid (optional).

    Returns:
    - meshgrid: Numpy array of release points (final lat/lon grid from which particles will be released).
    - release_grids: List of DataFrames, each containing release grid points for a specific species.
    - settlement_grids: List of DataFrames, each containing settlement grid points for a specific species or habitat.
    """
    # Extract the geojson file path
    file_path = spp_params['geojson_file_path']

    # Step 1: Create grid and mask data from hydrodynamic data
    full_grid, full_grid_df, coastal_grid, grid_ids = create_grid(coastalmask, lonarray, latarray)

    # Filter coastal grid using the polygon from GeoJSON
    _, shapely_polygon = read_geojson(file_path)
    coastal_grid_filtered = filter_by_polygon(coastal_grid, shapely_polygon)

    # Step 2: Ask user to choose particle release method
    print("Choose particle release method:")
    print("a) Release particles from all coastal areas.")
    print("b) Release particles from species distributions (GBIF data).")
    print("c) Upload your own species distribution data.")

    choice = input("Enter your choice (a, b, or c): ").strip().lower()

    # Step 3: Handle particle release method based on user choice
    if choice == 'a':
        print("Option a selected: Releasing particles from all coastal areas.")
        # Use all coastal grid points for particle release
        meshgrid = coastal_grid_filtered  # Use coastal grid as release points
        release_grids = None  # No release grids DataFrame needed for this option

    elif choice == 'b' or choice == 'c':
        release_grids = []  # Initialize a list to store DataFrames for each species

        if choice == 'b':
            print("Option b selected: Releasing particles from species distributions (GBIF data).")
            spp_occ_release = clean_spp
        else:
            print("Option c selected: Upload your species distribution data (CSV format expected).")
            file_path = input("Enter the input path and file name: ").strip()

            # Ensure the file has a .csv extension
            if not file_path.endswith(".csv"):
                file_path += ".csv"

            spp_occ_release = pd.read_csv(file_path)

        # Ensure species occurrence data has the necessary columns
        required_columns = {'latitude', 'longitude', 'species'}
        missing_columns = required_columns - set(spp_occ_release.columns)
        if missing_columns:
            raise ValueError(f"Species distribution data is missing the following columns: {', '.join(missing_columns)}")

        # Map species data to grid and create:
        # 1) a meshgrid of release points based on all species occurrence for the simulation
        # 2) release grids for post-processing
        meshgrid, release_grids = map_spp_to_grid(full_grid_df, spp_occ_release)

    else:
        raise ValueError("Invalid choice. Please enter 'a', 'b', or 'c'.")

    # Step 4: Ask user to choose particle settlement method
    print("Choose particle settlement method:")
    print("a) Settle at all coastal areas.")
    print("b) Settle based on species distributions (GBIF data).")
    print("c) Settle based on your own habitat suitability data.")

    choice = input("Enter your choice (a, b, or c): ").strip().lower()

    # Step 5: Handle particle settlement method based on user choice
    if choice == 'a':
        print("Option a selected: Settling particles at all coastal areas.")
        # Use coastal grid points for particle settlement
        settlement_grids = None  # No settlement grid DataFrame needed for this option

    elif choice == 'b' or choice == 'c':
        settlement_grids = []  # Initialize a list to store DataFrames for each species/habitat

        if choice == 'b':
            print("Option b selected: Settling particles based on species distributions (GBIF data).")
            spp_occ_settle = clean_spp
        else:
            print("Option c selected: Upload your habitat suitability data (CSV format expected).")
            file_path = input("Enter the input path and file name: ").strip()

            # Ensure the file has a .csv extension
            if not file_path.endswith(".csv"):
                file_path += ".csv"

            spp_occ_settle = pd.read_csv(file_path)

        # Ensure habitat suitability data has the necessary columns
        required_columns = {'latitude', 'longitude', 'species'}
        missing_columns = required_columns - set(spp_occ_settle.columns)
        if missing_columns:
            raise ValueError(f"Habitat suitability data is missing the following columns: {', '.join(missing_columns)}")

        # Map species data to grid and create settlement grids for post-processing
        _, settlement_grids = map_spp_to_grid(full_grid_df, spp_occ_settle)

    else:
        raise ValueError("Invalid choice. Please enter 'a', 'b', or 'c'.")

    # Step 6: Optionally save release and settlement grids as CSV
    save_grids(release_grids=release_grids, settlement_grids=settlement_grids, full_grid=full_grid)

    return meshgrid, release_grids, settlement_grids, full_grid_df, grid_ids
# endregion

# endregion

# region Creating the Fieldset and Components
## region Set Up Functions
### Unbeaching Field
def Unbeaching_Field(fieldset, landmask):
    # Create necessary components
    Lons = fieldset.U.lon
    Lats = fieldset.U.lat
    fieldmesh_x, fieldmesh_y = np.meshgrid(Lons, Lats)

    # Find indices of land and ocean
    oceancells = np.where(landmask == 0)
    landcells = np.where(landmask == 1)

    # Create empty arrays
    vectorfield_x = np.zeros(fieldmesh_x.shape)
    vectorfield_y = np.zeros(fieldmesh_y.shape)

    # Repeat the loop for all the land cells
    for i1 in range(len(landcells[1])):
        # Find the lon and lat for all of the land cells
        lon_coast = fieldmesh_x[landcells[0][i1], landcells[1][i1]]
        lat_coast = fieldmesh_y[landcells[0][i1], landcells[1][i1]]

        # Calculate the distance from each land cell to the ocean cells
        dist_lon = lon_coast - fieldmesh_x[oceancells[0], oceancells[1]]
        dist_lat = lat_coast - fieldmesh_y[oceancells[0], oceancells[1]]

        # Combine the values to get an array of the distances to ocean cells
        dist_to_ocean = np.sqrt(np.power(dist_lon, 2) + np.power(dist_lat, 2))

        # Calculate the minimum distance from each land cell to the ocean cells
        min_dist = np.min(dist_to_ocean)

        # Find the indices of the minimum distances within the overall distance array
        i_min_dist = np.where(dist_to_ocean == min_dist)

        # If there is only one ocean cell that is closest...
        if len(i_min_dist[0]) == 1:
            # Find the lon and lat of the ocean cell
            lon_ocean = fieldmesh_x[oceancells[0][i_min_dist[0][0]], oceancells[1][i_min_dist[0][0]]]
            lat_ocean = fieldmesh_y[oceancells[0][i_min_dist[0][0]], oceancells[1][i_min_dist[0][0]]]

            # Calculate the norm (avoid division by zero)
            norm = np.sqrt((lon_ocean - lon_coast)**2 + (lat_ocean - lat_coast)**2)
            if norm != 0:
                # Create a vector field
                vectorfield_x[landcells[0][i1], landcells[1][i1]] = (lon_ocean - lon_coast) / norm
                vectorfield_y[landcells[0][i1], landcells[1][i1]] = (lat_ocean - lat_coast) / norm

        # If there are multiple ocean cells that are closest...
        elif len(i_min_dist[0]) > 1:
            # Take the mean of the lons and lats of all the cells
            lon_ocean = np.mean(fieldmesh_x[oceancells[0][i_min_dist], oceancells[1][i_min_dist]])
            lat_ocean = np.mean(fieldmesh_y[oceancells[0][i_min_dist], oceancells[1][i_min_dist]])

            # Calculate the norm (avoid division by zero)
            norm = np.sqrt((lon_ocean - lon_coast)**2 + (lat_ocean - lat_coast)**2)
            if norm != 0:
                # Create a vector field
                vectorfield_x[landcells[0][i1], landcells[1][i1]] = (lon_ocean - lon_coast) / norm
                vectorfield_y[landcells[0][i1], landcells[1][i1]] = (lat_ocean - lat_coast) / norm

    return vectorfield_x, vectorfield_y, Lons, Lats

### Chunking
def calculate_chunk_size(hydrodata, available_memory, safety_factor=0.8):
    """
    Calculate optimal chunk sizes for a dataset to fit within available memory.

    Parameters:
    - hydrodata: xarray.Dataset object with dimensions.
    - available_memory: Available memory in MB.
    - safety_factor: Fraction of available memory to use for chunking.

    Returns:
    - Dictionary with optimal chunk sizes for each dimension.
    """
    # Get dataset dimensions
    dimensions = hydrodata.sizes
    time_size = dimensions.get('time', 1)
    lat_size = dimensions.get('latitude', 1)
    lon_size = dimensions.get('longitude', 1)
    depth_size = dimensions.get('depth', 1)  # Some datasets may not have depth

    # Determine data type size (assuming float32, which is 4 bytes per element)
    data_type_size = np.dtype('float32').itemsize

    # Calculate total number of elements in the dataset
    total_elements = time_size * lat_size * lon_size * depth_size

    # Initial guess for chunk sizes (for calculations only, not actual chunk sizes)
    initial_chunks = {'time': 10, 'lat': 50, 'lon': 50, 'depth': 10}

    # Calculate memory usage of one chunk based on initial chunk sizes
    chunk_size = (initial_chunks['time'] * initial_chunks['lat'] *
                  initial_chunks['lon'] * initial_chunks['depth'])
    chunk_memory_usage = chunk_size * data_type_size / (1024 ** 2)  # Convert to MB

    # Total memory available for chunks (after applying the safety factor)
    memory_for_chunks = available_memory * safety_factor

    # Calculate the number of chunks that fit within the available memory
    num_chunks = memory_for_chunks / chunk_memory_usage

    # Estimate optimal chunk sizes based on the number of chunks
    optimal_chunk_size = {
        'time': max(1, min(time_size, int(time_size / np.sqrt(num_chunks)))),
        'lat': max(1, min(lat_size, int(lat_size / np.sqrt(num_chunks)))),
        'lon': max(1, min(lon_size, int(lon_size / np.sqrt(num_chunks)))),
        'depth': max(1, min(depth_size, int(depth_size / np.sqrt(num_chunks))))
    }

    print(f"Calculated chunk sizes: {optimal_chunk_size}")

    return optimal_chunk_size
# endregion

## region Create Fieldset
def create_fieldset(hydro_data, landmask, grid_ids):
    """
    Create fieldset with dynamic chunking based on available memory.

    Parameters:
    - hydrodata: Path to the hydrodynamic data (e.g., .nc file or xarray)

    Returns:
    - FieldSet object with dynamically chunked data.
    """
    # Step 1: Check available system memory
    #mem = psutil.virtual_memory()
    #available_memory = mem.available / (1024 ** 2)  # Convert to MB

    # Step 2: Open dataset with dynamically calculated chunks
    hydro_data_copy = hydro_data.copy()

    # Calculate optimal chunk sizes based on available memory and dataset size
    # chunks = calculate_chunk_size(hydrodata_copy, available_memory)

    # Step 3: Proceed with fieldset creation (as in your previous code)
    var = hydro_data_copy.variables["uo"]
    num_dims = len(var.shape)

    if num_dims == 3:
        dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    elif num_dims == 4:
        dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time', 'depth': 'depth'}
    else:
        raise ValueError("Unsupported data dimensionality: {}".format(num_dims))

    variables = {"U": "uo", "V": "vo"}

    # Force garbage collection before creating the fieldset
    gc.collect()

    # Create the FieldSet (replace FieldSet creation logic based on your needs)
    fieldset = FieldSet.from_xarray_dataset(
        ds=hydro_data_copy,
        variables=variables,
        dimensions=dimensions,
        mesh='spherical'
    )

    # Step 4: Calculate all necessary components
    landvector_U, landvector_V, Lons, Lats = Unbeaching_Field(fieldset, landmask)

    # Step 5: Add Landmask
    fieldset.add_field(Field("landmask", landmask, lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat, mesh="spherical",
                             interp_method='nearest'))

    # Step 6: Add Grid IDs
    fieldset.add_field(Field("grid_ids", grid_ids, lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat, mesh="spherical",
                             interp_method='nearest'))

    # Step 7: Add unbeaching vector fields
    # Convert to Field objects
    U_land = Field('U_land', landvector_U, lon=Lons, lat=Lats, fieldtype='U', mesh='spherical',
                   interp_method='linear_invdist_land_tracer')
    V_land = Field('V_land', landvector_V, lon=Lons, lat=Lats, fieldtype='V', mesh='spherical',
                   interp_method='linear_invdist_land_tracer')

    # Add to the fieldset
    fieldset.add_field(U_land)
    fieldset.add_field(V_land)

    vectorfield_unbeaching = VectorField('UV_unbeach', U_land, V_land)  # Combine and convert to vector field

    fieldset.add_vector_field(vectorfield_unbeaching)  # Add to the fieldset

    return fieldset
# endregion

# endregion

# region Creating the ParticleSet
def create_pset(fieldset,meshgrid, runtime):
    """
    Creates a ParticleSet with staggered release intervals or a single release.

    Parameters:
    - fieldset: The fieldset object to be used for the particle set.
    - meshgrid: 2D numpy array with latitude and longitude pairs.
    - runtime: Total runtime of the simulation as a timedelta object.

    Returns:
    - pset: The created ParticleSet object.
    """
    # Extract lats and lons from the release meshgrid
    latarray = meshgrid[:, 1]  # Assuming the latitude is in the second column
    lonarray = meshgrid[:, 0]  # Assuming the longitude is in the first column

    # Flatten the latitude and longitude arrays
    lats = latarray.flatten()
    lons = lonarray.flatten()

    # Number of particles to release per site
    default_npart = 10
    npart_input = input(f"Enter the number of particles to release per site (default is {default_npart}): ")
    npart = int(npart_input) if npart_input else default_npart

    # Release interval
    release_mode = input("Enter 'staggered' for staggered release or 'single' for a single release: ").strip().lower()

    interval = None  # Initialize interval

    # Set up release times based on mode
    if release_mode == 'staggered':
        # Choose release interval (hourly, daily, weekly, or custom)
        default_interval = 'daily'
        release_interval = input(
            f"Enter the release interval ('hourly', 'daily', 'weekly' or a custom period in days (e.g., 3 for a 3-day interval), default is {default_interval}): ").strip().lower()

        if release_interval == 'hourly':
            interval = timedelta(hours=1)
        elif release_interval == 'daily':
            interval = timedelta(days=1)
        elif release_interval == 'weekly':
            interval = timedelta(weeks=1)
        else:
            try:
                # Assume the user inputs a number of days for a custom period
                custom_days = int(release_interval)
                interval = timedelta(days=custom_days)
            except ValueError:
                raise ValueError("Invalid release interval. Please enter 'hourly', 'daily', 'weekly', or a custom number of days (e.g., 3 for a 3-day interval).")
        # Ensure runtime is a numeric value in days (convert timedelta to days)
        #if isinstance(runtime, timedelta):
        #    runtime_seconds = runtime.total_seconds()  # Convert timedelta to seconds
        #else:
        #    raise ValueError("Runtime should be provided as a timedelta object.")

        # Convert interval to seconds
        #interval_seconds = interval.total_seconds()  # Convert interval to seconds

        # Step 5: Calculate how many intervals fit into the runtime
        num_intervals = math.ceil(runtime / interval)  # Round up the number of intervals
        adjusted_runtime = num_intervals * interval  # Adjust the runtime to fit intervals
        #runtime_days = int(adjusted_runtime / 86400)

        # Ensure there are enough particles for the number of intervals
        #if npart / num_intervals < 1:
         #   print(f"Warning: You have fewer particles than intervals ({npart} < {num_intervals}). Adjusting npart.")

        # Adjust npart to ensure we can release whole particles
        particles_per_interval = math.ceil(npart / num_intervals)  # Round up particles per interval
        total_particles = particles_per_interval * num_intervals  # Calculate the total number of particles

        # Update npart if it needs to be increased
        if particles_per_interval != npart:
            npart = total_particles
            print(
                f"Adjusted npart to {npart} to fit the number of intervals and ensure whole particles per release.")

        # Calculate particles per interval and print result
        print(
            f"Staggered release chosen. {total_particles} particles will be released with {particles_per_interval} particles per location across {num_intervals} intervals.")

        # Calculate the repeat interval (repeatdt)
        repeatdt = adjusted_runtime / num_intervals  # Adjusted interval duration
        print(f"The new runtime is {runtime_days} days to fit {num_intervals} intervals.")
        print(f"Particles will be released every {repeatdt} days.")
    elif release_mode == 'single':
        repeatdt = None
        adjusted_runtime = int(runtime)
    else:
        raise ValueError("Invalid release mode. Please enter 'staggered' or 'single'.")

    # Particle Class with extra variables
    extra_vars = [
        Variable("distance", initial=0.0, dtype=np.float32),
        Variable("prev_lon", dtype=np.float32, to_write=False, initial=attrgetter("lon")),
        Variable("prev_lat", dtype=np.float32, to_write=False, initial=attrgetter("lat")),
        Variable('on_land', dtype=np.int32, initial=0),
        # Variable('release_id', dtype=np.int32, initial=0),  # Track release cohort,
        Variable('grid_id', dtype=np.int32, initial=0)
    ]

    # Create ParticleClass
    Particle = JITParticle.add_variables(extra_vars)

    # Include npart - number of particles to be released at each site
    lat = list(itertools.chain.from_iterable(itertools.repeat(x, npart) for x in lats))
    lon = list(itertools.chain.from_iterable(itertools.repeat(x, npart) for x in lons))

    # Set constant horizontal diffusivity (kh) in m^2/s
    default_kh = 100
    kh_input = input(f"Enter the constant horizontal diffusivity (kh) in m^2/s (default is {default_kh}): ")
    kh = float(kh_input) if kh_input else default_kh
    fieldset.add_constant_field("Kh_zonal", kh, mesh="spherical")
    fieldset.add_constant_field('Kh_meridional', kh, mesh='spherical')

    # Create and return the ParticleSet based on the release mode
    pset = ParticleSet(fieldset=fieldset,
                       pclass=Particle,
                       lon=lon,
                       lat=lat,
                       repeatdt = repeatdt)

    return pset, adjusted_runtime

# endregion

# region Creating Custom Kernels
def create_custom_kernels():
    """
    Function to create and return all custom kernels for particle behavior.
    """

    def DeleteErrorParticle(particle, fieldset, time):
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()

    def TotalDistance(particle, fieldset, time):
        """Calculate the distance travelled by the particle in latitude and longitude."""
        lat_dist = (particle.lat - particle.prev_lat) * 1.11e2
        lon_dist = (
                (particle.lon - particle.prev_lon)
                * 1.11e2
                * math.cos(particle.lat * math.pi / 180)
        )
        # Calculate the total Euclidean distance travelled by the particle
        particle.distance += math.sqrt(math.pow(lon_dist, 2) + math.pow(lat_dist, 2))

        # Set the stored values for the next iteration
        particle.prev_lon = particle.lon
        particle.prev_lat = particle.lat

    def Sample_land(particle, fieldset, time):
        """Sample if the particle is on land."""
        particle.on_land = fieldset.landmask[time, particle.depth, particle.lat, particle.lon]

    def Grid_id(particle, fieldset, time):
        """Sample if the particle is on land."""
        particle.grid_id = fieldset.grid_ids[time, particle.depth, particle.lat, particle.lon]

    def Unbeaching(particle, fieldset, time):
        """Update particle position if it is on land."""
        if particle.on_land == 1:  # If the particle is on land
            (u_land, v_land) = fieldset.UV_unbeach[
                time, particle.depth, particle.lat, particle.lon]  # land U and V velocities
            particle_dlon += u_land * particle.dt  # Update lon with land U velocity
            particle_dlat += v_land * particle.dt  # Update lat with land V velocity
            particle.on_land = 0  # The particle is no longer on land

    # Return the kernels in the desired format
    return (
            pset.Kernel(AdvectionRK4) +
            pset.Kernel(TotalDistance) +
            pset.Kernel(DiffusionUniformKh) +
            pset.Kernel(Sample_land) +
            pset.Kernel(Unbeaching) +
            pset.Kernel(Grid_id)+
            pset.Kernel(DeleteErrorParticle)
    )
# endregion

# region Executing the Simulation
def execute_simulation(pset, kernels, adjusted_runtime):
    """
    Optimized function to execute the particle simulation, with parallelization, chunking,
    and efficient handling of large outputs using Dask.
    """
    # Step 2: Prompt the user for output file path (without extension)
    output_path = input("Enter the output path and file name to save the final simulation output: ")

    # Ensure that the file has a .nc extension
    if not output_path.endswith(".zarr"):
        output_path += ".zarr"

    # Step 3: Prompt the user for output frequency
    output_dt = int(input("Enter output frequency in hours: "))
    output_dt = timedelta(hours=output_dt)

    # Step 4: Prompt the user for dt in minutes
    dt_minutes = int(input("Enter time step (dt) in minutes: "))
    dt = timedelta(minutes=dt_minutes)

    # Step 5: Runtime formatting
    runtime = timedelta(days=adjusted_runtime)

    # Step 5: Set chunk sizes based on the number of particles for better efficiency
    #num_particles = len(pset)
    #chunk_size = (min(1e4, num_particles), 1)  # Adjust based on particle count
    # manual chunks=(int(1e4), 1)

    # Step 6: Create a ParticleFile object with chunking and output frequency
    output_file = pset.ParticleFile(
        name=output_path,
        outputdt=output_dt,
        chunks=(int(1e4), 1)  # Efficient chunking scheme for large datasets
    )

    # Step 9: Execute the simulation with parallelization
    print("Running the simulation...")
    pset.execute(
        kernels,
        runtime=runtime,
        dt=dt,
        output_file=output_file
    )

    # Step 10: Automatically load the output using xarray's open_zarr
    print("Loading the output into the environment...")
    ds = xr.open_zarr(output_path)

    # Convert the .zarr file to a dataframe
    traj = ds.to_dataframe()
    traj.reset_index(inplace=True)

    # Assign release_id based on the first observation time for each trajectory
    # Step 1: Identify the first observation time for each trajectory
    #first_obs = traj[traj['obs'] == 0][['trajectory', 'time']].copy()

    # Step 2: Create a mapping of first observation time to release_id
   # first_obs['release_id'] = first_obs.groupby('time').ngroup()

    # Step 3: Merge the release_id back to the original DataFrame
    #traj = traj.merge(first_obs[['trajectory', 'release_id']], on=['trajectory', 'time'], how='left')

    # Return the DataFrame for further use
    return traj

# endregion

def assign_release_id(traj):
    """
    Assigns a release_id to each trajectory based on the first observation time.

    Parameters:
    - traj: DataFrame containing trajectory data with columns 'trajectory', 'time', and 'obs'.

    Returns:
    - traj: Updated DataFrame with a new 'release_id' column.
    """
    # Step 1: Identify the first observation time for each trajectory
    first_obs = traj[traj['obs'] == 0][['trajectory', 'time']].copy()

    # Step 2: Create a mapping of first observation time to release_id
    first_obs['release_id'] = first_obs.groupby('time').ngroup()

    # Step 3: Merge the release_id back to the original DataFrame
    traj = traj.merge(first_obs[['trajectory', 'release_id']], on=['trajectory', 'time'], how='left')

    return traj2

# region Post-Processing
## Seasonality
def spawning_season_filter(traj, traits_df):
    # Ensure the 'time' column in traj is in datetime format
    traj['time'] = pd.to_datetime(traj['time'])

    # Convert the spawning_start and spawning_end columns in traits to datetime
    traits_df['spawning_start'] = pd.to_datetime(traits_df['spawning_start'], dayfirst=True)
    traits_df['spawning_end'] = pd.to_datetime(traits_df['spawning_end'], dayfirst=True)

    # Dictionary to store the resulting DataFrames
    spawning_dfs = {}

    # Loop through each species in the traits DataFrame
    for _, row in traits_df.iterrows():
        species_name = row['species_name']
        start_date = row['spawning_start']
        end_date = row['spawning_end']

        # List to hold trajectories that meet the criteria
        valid_trajectories = []

        # Iterate over unique trajectory IDs
        for traj_id in traj['trajectory'].unique():
            # Subset the data for this trajectory
            traj_subset = traj[traj['trajectory'] == traj_id]

            # Find the row where obs == 0 (spawning point)
            spawning_point = traj_subset[traj_subset['obs'] == 0]

            if not spawning_point.empty:
                # Get the spawning time (time of obs == 0)
                spawning_time = spawning_point.iloc[0]['time']

                # Check if the spawning time falls within the species' spawning season
                if start_date <= spawning_time <= end_date:
                    valid_trajectories.append(traj_subset)

        # Concatenate all valid trajectories for the species into a single DataFrame
        if valid_trajectories:
            spawning_dfs[species_name] = pd.concat(valid_trajectories)
        else:
            spawning_dfs[species_name] = pd.DataFrame()  # Empty DataFrame if no matches

    return spawning_dfs

def spawning_season_filter2(traj, traits):
    # Ensure the 'time' column is in datetime format
    traj['time'] = pd.to_datetime(traj['time'])

    # Extract the month from the 'time' column
    traj['month'] = traj['time'].dt.month

    # Dictionary to store the resulting DataFrames
    spawning_dfs = {}

    # Dictionary to convert month names to numbers
    months_dict = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    # Loop through each species in the traits DataFrame
    for _, row in traits.iterrows():
        species_name = row['species_name']
        start_month = months_dict[row['spawning_start']]
        end_month = months_dict[row['spawning_end']]

        # List to hold trajectories that meet the criteria
        valid_trajectories = []

        # Iterate over unique trajectory IDs
        for traj_id in traj['trajectory'].unique():
            # Subset the data for this trajectory
            traj_subset = traj[traj['trajectory'] == traj_id]

            # Find the row where obs == 0 (spawning point)
            spawning_point = traj_subset[traj_subset['obs'] == 0]

            if not spawning_point.empty:
                # Get the spawning time (time of obs == 0)
                spawning_time = spawning_point.iloc[0]['time']
                spawning_month = spawning_time.month

                # Check if the spawning time falls within the species' spawning season
                if start_month <= end_month:
                    # Simple case: spawning season within the same year
                    if start_month <= spawning_month <= end_month:
                        valid_trajectories.append(traj_subset)
                else:
                    # Spawning season wraps around the end of the year
                    if spawning_month >= start_month or spawning_month <= end_month:
                        valid_trajectories.append(traj_subset)

        # Concatenate all valid trajectories for the species into a single DataFrame
        if valid_trajectories:
            spawning_dfs[species_name] = pd.concat(valid_trajectories)
        else:
            spawning_dfs[species_name] = pd.DataFrame()  # Empty DataFrame if no matches

    return spawning_dfs

## Planktonic Larval Duration
def PLD_filter(spawning_dfs, traits_df):
    """
    Crop each species dataframe to retain only observations within the max_PL duration.

    Parameters:
    - species_dfs: Dictionary with species names as keys and DataFrames as values.
    - traits_df: DataFrame containing species traits, including max_PLD.

    Returns:
    - Dictionary with cropped species dataframes.
    """
    cropped_dfs = {}

    for species_name, df in spawning_dfs.items():
        # Retrieve the max_PLD for the current species
        max_pld = traits_df.loc[traits_df['species_name'] == species_name, 'max_PLD'].values[0]

        # Ensure the 'time' column is in datetime format
        df['time'] = pd.to_datetime(df['time'])

        # Get unique trajectories
        trajectories = df['trajectory'].unique()

        # Dictionary to hold cropped data for this species
        cropped_data = []

        for traj in trajectories:
            traj_df = df[df['trajectory'] == traj]

            # Check if there are any observations for this trajectory
            if not traj_df.empty:
                # Get the time of the first observation (obs = 0)
                start_time = traj_df['time'].min()

                # Calculate the time window based on max_PL duration
                end_time = start_time + pd.to_timedelta(max_pld, unit='D')

                # Filter observations within the time window
                within_window = traj_df['time'].between(start_time, end_time)

                # Append filtered data
                cropped_data.append(traj_df[within_window])

        # Concatenate all cropped data for this species
        if cropped_data:
            cropped_dfs[species_name] = pd.concat(cropped_data)

    return cropped_dfs

## Species-specific Release and Settlement
def release_settle_filter(cropped_dfs, release_grids, settlement_grids, grid_ids):
    """
    Filters particle trajectories to keep only those starting at release points where the species occur and
    ending at settlement points (with an optional buffer around the settlement points).

    Parameters:
    trajectories_dict (dict): Dictionary of trajectory data for each species.
    occurrence_data (dict): Dictionary of occurrence data for each species (release locations).
    settlement_data (dict): Dictionary of settlement data for each species.
    full_grid_df (pd.DataFrame): Dataframe containing grid_id, latitude, and longitude for the entire study area.
    grid_matrix (np.array): Matrix of grid_ids for the entire study area.

    Returns:
    dict: Filtered trajectory data for each species, retaining full trajectories that meet start and end conditions.
    """

    # Ask the user for the buffer size
    while True:
        try:
            buffer = int(input("Enter the buffer size (number of adjacent grid cells around the settlement points): "))
            if buffer >= 0:
                break
            else:
                print("Please enter a non-negative integer for the buffer size.")
        except ValueError:
            print("Invalid input. Please enter an integer for the buffer size.")

    def get_adjacent_grids(grid_id, grid_ids, buffer):
        """ Get the neighboring grid cells around a given grid_id, respecting the matrix structure. """
        # Find the row and column in the matrix
        position = np.where(grid_ids == grid_id)
        if len(position[0]) == 0:
            return []
        row, col = position[0][0], position[1][0]

        # Get adjacent cells within the buffer (up/down/left/right diagonals)
        adjacent_grids = []
        for i in range(-buffer, buffer + 1):
            for j in range(-buffer, buffer + 1):
                new_row, new_col = row + i, col + j
                if 0 <= new_row < grid_ids.shape[0] and 0 <= new_col < grid_ids.shape[1]:
                    adjacent_grids.append(grid_ids[new_row, new_col])

        return adjacent_grids

    # Filtered trajectories will be stored here
    filtered_trajectories = {}

    # Iterate through each species
    for species, traj_df in cropped_dfs.items():
        # 1. Filter based on release location (grid_id in occurrence data)
        occurrence_grids = set(release_grids[species]['grid_id'])

        # 2. Filter based on settlement location (grid_id in settlement data, with buffer)
        settlement = set(settlement_grids[species]['grid_id'])

        # Expand the settlement grids with a buffer
        buffered_settlement_grids = set()
        for grid_id in settlement:
            buffered_settlement_grids.update(get_adjacent_grids(grid_id, grid_ids, buffer))

        # List to store filtered trajectories for the species
        filtered_trajectories_species = []

        # Group by trajectory to process each one individually
        grouped = traj_df.groupby('trajectory')

        for trajectory_id, group in grouped:
            # Ensure the group is sorted by 'obs'
            group = group.sort_values(by='obs')

            # Identify the first and last observation
            first_obs_grid_id = group.iloc[0]['grid_id']  # First time step (obs=0)
            last_obs_grid_id = group.iloc[-1]['grid_id']  # Last time step (obs=max)

            # Check if first observation falls in occurrence grids
            first_in_occurrence = first_obs_grid_id in occurrence_grids

            # Check if last observation falls in buffered settlement grids
            last_in_settlement = last_obs_grid_id in buffered_settlement_grids

            # If both conditions are met, retain the full trajectory
            if first_in_occurrence and last_in_settlement:
                filtered_trajectories_species.append(group)

        # If any trajectories pass the filter, save them for the species
        if filtered_trajectories_species:
            # Concatenate all filtered trajectories for this species and store in dictionary
            filtered_trajectories[species] = pd.concat(filtered_trajectories_species)

    return filtered_trajectories, buffered_settlement_grids

# Complete Post-Processing
def post_process(traj, traits, release_grids, settlement_grids, grid_ids):
    """
    Perform post-processing on particle trajectory data by applying filters such as PLD, seasonality,
    and spawning/settlement filtering in sequence.

    Args:
    - traj (pd.DataFrame): Trajectory data.
    - traits (pd.DataFrame): Dataframe containing species traits information.
    - occurrence_data (dict): Dictionary of occurrence data for species.
    - settlement_data (dict): Dictionary of settlement data for species.
    - full_grid_df (pd.DataFrame): Dataframe of grid_id, lat/lon for the study area.
    - grid_ids (np.array): Matrix of grid_ids for spatial relationships.

    Returns:
    - processed_data: The trajectory data after all post-processing filters have been applied.
    - buffered_settlement_grids (optional): Grids used in the release/settlement filtering, if applicable.
    """

    # Start with a copy of the original trajectory data
    data = traj.copy()

    # Change format of traits list to dataframe
    traits_df = pd.DataFrame(traits)

    # Step 1: Apply the spawning seasonality filter
    print("Applying spawning seasonality filter...")
    spawning_dfs = spawning_season_filter(data, traits_df)

    # Step 2: Apply the PLD (Planktonic Larval Duration) filter
    print("Applying PLD filter...")
    cropped_dfs = PLD_filter(spawning_dfs, traits_df)

    # Step 3: Apply the spawning/settlement site filter
    print("Applying release/settlement site filter...")
    processed_data, buffered_settlement_grids = release_settle_filter(cropped_dfs, release_grids, settlement_grids, grid_ids)

    # Return the final processed data and buffered settlement grids
    return processed_data, buffered_settlement_grids

# endregion

# region Post-Analysis
##Retention Index
def retention_index(filtered_trajectories):
    species_retention = {}
    overall_same_grid_count = 0
    overall_total_trajectories = 0

    for species, df in filtered_trajectories.items():
        # Skip if the dataframe is empty
        if df.empty:
            print(f"No data for {species}, skipping retention index calculation.")
            continue

        # Filter the start and end points (obs == 0 for start and obs == max(obs) for end)
        start_df = df[df['obs'] == 0]
        end_df = df.loc[df.groupby('trajectory')['obs'].idxmax()]

        # Merge the start and end dataframes on the trajectory ID
        merged_df = pd.merge(start_df, end_df, on='trajectory', suffixes=('_start', '_end'))
        #think there is an issue here as retention indices are too high

        # Count the number of trajectories that started and ended in the same grid cell
        same_grid_count = (merged_df['grid_id_start'] == merged_df['grid_id_end']).sum()

        # Calculate the total number of trajectories for this species
        total_trajectories = len(end_df)

        # Calculate the species-level retention percentage
        species_retention_percentage = (same_grid_count / total_trajectories) * 100
        species_retention[species] = species_retention_percentage

        # Add to overall counts for the overall retention calculation
        overall_same_grid_count += same_grid_count
        overall_total_trajectories += total_trajectories

    if overall_total_trajectories > 0:
        # Calculate overall retention percentage across all species
        overall_retention_percentage = (overall_same_grid_count / overall_total_trajectories) * 100
    else:
        overall_retention_percentage = 0

    return species_retention, overall_retention_percentage #merged_df

## Mortality
def mortality(filtered_trajectories):
    species_mortality = {}
    overall_start_count = 0
    overall_successful_count = 0

    for species, df in filtered_trajectories.items():
        # Skip if the dataframe is empty
        if df.empty:
            print(f"No data for {species}, skipping mortality calculation.")
            continue

        # Filter the start and end points (obs == 0 for start and obs == max(obs) for end)
        start_df = df[df['obs'] == 0]
        end_df = df.loc[df.groupby('trajectory')['obs'].idxmax()]

        # Calculate the total number of starting trajectories for this species
        total_trajectories = len(start_df)

        # Calculate the number of successful trajectories (those that reached the end)
        successful_trajectories = len(end_df)

        # Calculate the species-level mortality percentage
        mortality_percentage = (1 - (successful_trajectories / total_trajectories)) * 100
        species_mortality[species] = mortality_percentage

        # Add to overall counts for the overall mortality calculation
        overall_start_count += total_trajectories
        overall_successful_count += successful_trajectories

    if overall_start_count > 0:
        # Calculate overall mortality percentage across all species
        overall_mortality_percentage = (1 - (overall_successful_count / overall_start_count)) * 100
    else:
        overall_mortality_percentage = 0

    return species_mortality, overall_mortality_percentage

## Network Analysis
def network_analysis(filtered_trajectories):
    species_connections = {}
    top_10_overall = []
    bottom_10_overall = []

    for species, df in filtered_trajectories.items():
        # Skip if the dataframe is empty
        if df.empty:
            print(f"No data for {species}, skipping network analysis.")
            continue

        # Count the number of unique connections per grid cell (id_end)
        connection_counts = df.groupby('grid_id').size().reset_index(name='connections')

        # Identify the top and bottom 10 grid cells by number of connections for the current species
        top_10 = connection_counts.nlargest(10, 'connections')
        bottom_10 = connection_counts.nsmallest(10, 'connections')

        # Store the connection counts, top 10, and bottom 10 in the species_connections dictionary
        species_connections[species] = {
            'connection_counts': connection_counts,
            'top_10': top_10,
            'bottom_10': bottom_10
        }

        # Append top and bottom 10 to the overall list
        top_10_overall.append(top_10)
        bottom_10_overall.append(bottom_10)

    if top_10_overall and bottom_10_overall:
        # Combine the overall top and bottom 10 across all species into a single dataframe
        top_10_overall_df = pd.concat(top_10_overall).nlargest(10, 'connections')
        bottom_10_overall_df = pd.concat(bottom_10_overall).nsmallest(10, 'connections')
    else:
        top_10_overall_df = pd.DataFrame()
        bottom_10_overall_df = pd.DataFrame()

    return species_connections, top_10_overall_df, bottom_10_overall_df

## Distance Travelled
def calculate_total_distance_per_species(filtered_trajectories):
    total_distances = []

    for species, df in filtered_trajectories.items():
        # Skip if the dataframe is empty
        if df.empty:
            print(f"No data for {species}, skipping distance calculation.")
            continue

        # Sum the distance traveled for the current species
        total_distance = df['distance'].sum()

        # Store the result in a dictionary with the species name and total distance
        total_distances.append({'species': species, 'total_distance': total_distance})

    # Convert the list of dictionaries into a DataFrame for better readability
    total_distance_df = pd.DataFrame(total_distances)

    return total_distance_df

# region Complete Post-Analysis
def post_analysis(processed_data):
    """
    Perform post-analysis on the processed trajectory data by interactively selecting analyses such as
    Retention Index, Mortality, Network Analysis, and Distance Travelled.

    Args:
    - processed_data (dict): The output data from the post-processing stage for each species.

    Returns:
    - Dictionary containing the results of selected analyses.
    """

    # Dictionary to hold analysis results
    analysis_results = {}

    # User selects which analyses to apply
    apply_retention_index = input("Do you want to calculate the Retention Index? (yes/no): ").strip().lower()
    apply_mortality = input("Do you want to calculate Mortality? (yes/no): ").strip().lower()
    apply_network_analysis = input("Do you want to perform a Network Analysis? (yes/no): ").strip().lower()
    apply_distance_travelled = input("Do you want to assess the distance travelled? (yes/no): ").strip().lower()

    # Retention Index
    if apply_retention_index in ['yes', 'y']:
        print("Calculating Retention Index...")
        species_retention, overall_retention, merged_df = retention_index(processed_data)
        analysis_results['retention_index'] = {
            'species_retention': species_retention,
            'overall_retention': overall_retention,
            'merged_df': merged_df
        }

    # Mortality
    if apply_mortality in ['yes', 'y']:
        print("Calculating Mortality...")
        species_mortality, overall_mortality = mortality(processed_data)
        analysis_results['mortality'] = {
            'species_mortality': species_mortality,
            'overall_mortality': overall_mortality
        }

    # Network Analysis
    if apply_network_analysis in ['yes', 'y']:
        print("Performing Network Analysis...")
        species_connections, top_10_overall, bottom_10_overall = network_analysis(processed_data)
        analysis_results['network_analysis'] = {
            'species_connections': species_connections,
            'top_10_overall': top_10_overall,
            'bottom_10_overall': bottom_10_overall
        }

    # Distance Travelled
    if apply_distance_travelled in ['yes', 'y']:
        print("Calculating Distance Travelled...")
        total_distance_df = calculate_total_distance_per_species(processed_data)
        analysis_results['distance_travelled'] = total_distance_df

    return analysis_results
# endregion

# endregion

# region Complete Interactive Workflow
def workflow_step(step_number):
    """
    Main function to guide the user through the interactive workflow.
    This will call all necessary functions in sequence based on user inputs.
    The user will have to specify the step number of the workflow as they work their way through.
    """

    print("\nStarting the interactive workflow...")

    # Load Modules
    load_modules()

    # Declare Global variables
    global runtime, species_traits_path, traits, hydro_params, hydro_data
    global clean_spp, spp_params, landmask, coastalmask, lonarray, latarray, traits
    global meshgrid, release_grids, settlement_grids, full_grid_df, grid_ids
    global fieldset, adjusted_runtime, pset
    global kernels
    global traj
    global filtered_trajectories, buffered_settlement_grids

    if step_number == 1:
        # Step 2: Collect Hydrodynamic Parameters
        print("\nStep 1: Select Hydrodynamic Data")
        print("This package uses hydrodynamic data from the Copernicus Marine Service.")
        print("Use the link below to set up a free account to browse and select your hydrodynamic data.")
        print("To access the data, you will need information such as your username, password, dataset ID number and longitude and latitude ranges of your study area.")
        print("This package requires the user to provide a species traits dataset with information on the planktonic larval duration and timing of spawning of each species.")
        print("This data will then be used to select the time range of the simulation and hydrodynamic data.")
        print("Please see Section X in the User Manual for more information.")
        print("https://data.marine.copernicus.eu/products")

        # Collect parameters
        hydro_params, species_traits_path, runtime, traits = params()  # This will call the existing params() function

        # Download Hydrodynamic Data
        hydro_data = download_hydrodynamic_data(**hydro_params)

        print("Step 1 is completed. Run the following in the next cell to continue:\nworkflow_step(2)")

    elif step_number == 2:
        # Step 4: Collect Species Occurrence Data Parameters
        print("\nStep 2: Download Species Occurrence Data")
        print("This package uses species occurrence data to determine species-specific release and settlement areas.")
        print("You have the option to download species occurrence data from the Global Biodiversity Information System (GBIF) using the species contained in the trait dataset uploaded in Step 1 or to provide your own species occurrence data.")
        print("Please visit the GBIF website for further information: https://www.gbif.org/ ")
        print("Please see Section X in the User Manual for more information.")

        # Call spp_occ to gather species occurrence data
        clean_spp, spp_params, landmask, coastalmask, lonarray, latarray = spp_occ(hydro_data, species_traits_path)

        print("Step 2 is completed. Run the following in the next cell to continue:\nworkflow_step(3)")

    elif step_number == 3:
        # Step 5: Grid creation and release and settlement site setup
        print("\nStep 3: Select a larval release and settlement strategy.")
        print("This package has the option to release larvae from all coastal sites or from known species-specific distributions.")
        print("Please see Section X in the User Manual for more information.")

        # Set up release and settlement grids
        meshgrid, release_grids, settlement_grids, full_grid_df, grid_ids = setup_release_settlement(coastalmask, clean_spp, spp_params, lonarray, latarray)

        # Create Fieldset from hydrodynamic data
        fieldset = create_fieldset(hydro_data, landmask, grid_ids)

        print("Release and settlement grid setup complete.")
        print("Step 3 is completed. Run the following in the next cell to continue:\nworkflow_step(4)")

    elif step_number == 4:
        # Step 7: Create ParticleSet
        print("\nStep 4: Create a ParticleSet for your community.")
        print("This package allows manual configuration of particles released per site, staggered or single release options, release frequency, and horizontal diffusivity.")
        print("Please see Section X in the User Manual for more information.")

        # Create ParticleSet
        pset, adjusted_runtime = create_pset(fieldset, meshgrid, runtime)

        # Create custom kernels
        kernels = create_custom_kernels()

        print("Step 4 is completed. Run the following in the next cell to continue:\nworkflow_step(5)")

    elif step_number == 5:
        # Step 9: Execute Simulation
        print("\nStep 5: Execute the simulation.")
        print("Please see Section X in the User Manual for more information.")

        # Execute simulation
        execute_simulation(pset, kernels, adjusted_runtime)

        # Post-process trajectories
        filtered_trajectories, buffered_settlement_grids = post_process(traj, traits, release_grids, settlement_grids, grid_ids)

        print("Step 5 is completed. Run the following in the next cell to continue:\nworkflow_step(6)")

    elif step_number == 6:
        # Step 11: Post-Analysis
        print("\nStep 6: Perform Post-Analysis.")
        print("This package can estimate standard connectivity parameters, such as centrality measures, retention index, mortality, or distance traveled for each species.")
        print("It can also run specific community analyses, such as network analysis, propagule pressure estimation, and community similarity analysis.")
        print("Please see Section X in the User Manual for more information.")

        # Perform post-analysis
        post_analysis(filtered_trajectories)

        # Indicate completion of the workflow
        print("\nInteractive workflow completed!")
# endregion

# endregion