# region Setting the Hydrodynamic Parameters
## Collecting Hydrodynamic Parameters
def collect_parameters():
    """
    Collects the hydrodynamic data parameters from the user.

    Returns:
        dict: Dictionary containing the user-provided parameters.
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
    time_range = [
        input("Enter the start date (YYYY-MM-DD): "),
        input("Enter the end date (YYYY-MM-DD): ")
    ]
    output_path = input("Enter the output path (including filename): ")

    # Prompt the user for export option
    export_input = input("Do you want to export the data to a NetCDF file? (yes/no): ").strip().lower()
    export = True if export_input in ["yes", "y"] else False

    return {
        "dataset_id": dataset_id,
        "longitude_range": longitude_range,
        "latitude_range": latitude_range,
        "time_range": time_range,
        "username": username,
        "password": password,
        "output_path": output_path,
        "export": export
    }

## Edit Hydrodynamic Parameters
def edit_parameters(params):
    """
    Allows the user to edit existing parameters.

    Args:
        params (dict): Dictionary of existing parameters.

    Returns:
        dict: Updated dictionary of parameters.
    """
    # Define a list of keys that are lists and their expected types
    list_keys = {
        'longitude_range': float,
        'latitude_range': float,
        'time_range': str  # Time range will stay as a string
    }

    # Iterate over each parameter in the dictionary
    for key, value in params.items():
        edit = input("Do you want to change {key} (current value: {value})? (yes/no): ".format(key=key, value=value)).strip().lower()

        if edit in ["yes", "y"]:
            if key in list_keys:
                # Handle list values separately (longitude_range, latitude_range, time_range)
                if len(value) == 2:
                    if key == 'time_range':
                        # Handle date input for time_range
                        try:
                            new_value = [
                                input("Enter the new start date for {key} (YYYY-MM-DD) (current value: {start}): ".format(key=key, start=value[0])),
                                input("Enter the new end date for {key} (YYYY-MM-DD) (current value: {end}): ".format(key=key, end=value[1]))
                            ]
                            # Validate date format
                            for date_str in new_value:
                                datetime.strptime(date_str, '%Y-%m-%d')
                        except ValueError:
                            print("Invalid date format. Please enter dates in YYYY-MM-DD format.")
                            continue
                    else:
                        # Handle numeric values for latitude_range, longitude_range
                        try:
                            new_value = [
                                list_keys[key](input(
                                    "Enter the new value for {key} - first element (current value: {first}): ".format(key=key, first=value[0]))),
                                list_keys[key](input(
                                    "Enter the new value for {key} - second element (current value: {second}): ".format(key=key, second=value[1])))
                            ]
                        except ValueError:
                            print("Invalid input. Please enter values of type {type_name}.".format(type_name=list_keys[key].__name__))
                            continue
                else:
                    print("Expected a list with 2 elements for {key}.".format(key=key))
                    continue
            else:
                # For non-list values, just update the parameter
                new_value = input("Enter the new value for {key} (current value: {value}): ".format(key=key, value=value))
                # Try to cast numeric values (if possible) to their original types
                try:
                    if isinstance(value, bool):
                        new_value = new_value.lower() in ['true', 'yes', '1']
                    elif isinstance(value, (int, float)):
                        new_value = type(value)(new_value)
                except ValueError:
                    print("Invalid input. Keeping the original value for {key}.".format(key=key))
                    continue

            # Update the parameter in the dictionary
            params[key] = new_value

    # Save the updated parameters to the JSON file
    with open("hydrodynamic_parameters.json", "w") as file:
        json.dump(params, file, indent=4)

    return params

## Load Hydrodynamic Parameters
def parameters():
    """
    Prompts the user to input hydrodynamic data parameters, allows for reusing or editing existing parameters,
    and then saves them to a JSON file.

    Returns:
        dict: Dictionary containing all the user-provided parameters.
    """
    params_file = "hydrodynamic_parameters.json"

    # Check if parameters file exists and offer to reuse
    if os.path.exists(params_file):
        reuse_input = input("Found existing parameters. Do you want to reuse them? (yes/no): ").strip().lower()
        if reuse_input in ["yes", "y"]:
            with open(params_file, "r") as file:
                params = json.load(file)
                print("Reusing existing parameters:")
                for key, value in params.items():
                    print(f"{key}: {value}")

            # Offer to edit specific parameters
            edit_input = input("Do you want to edit any parameters? (yes/no): ").strip().lower()
            if edit_input in ["yes", "y"]:
                params = edit_parameters(params)

            # Save and return the (possibly edited) parameters
            with open(params_file, "w") as file:
                json.dump(params, file)
            return params

    # If not reusing or no existing file, prompt the user for input
    params = collect_parameters()

    # Offer to review and edit before saving
    review_input = input("Do you want to review and edit the parameters before saving? (yes/no): ").strip().lower()
    if review_input in ["yes", "y"]:
        params = edit_parameters(params)

    # Save the parameters to a JSON file
    with open(params_file, "w") as file:
        json.dump(params, file)

    print("Parameters have been saved to 'hydrodynamic_parameters.json'.")
    return params
# endregion

# region Downloading Hydrodynamic Parameters
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
    import copernicusmarine as cm

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
        hydrodata = xr.open_dataset(output_path)
    else:
        # Directly use the data without saving to a file
        hydrodata = data

    return hydrodata
# endregion

# region Downloading Species Occurrence Data

# endregion
