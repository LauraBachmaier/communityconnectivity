def welcome_message():
    print("Welcome to the Community Connectivity Package!")
    print("This package facilitates multi-species dispersal models using Ocean Parcels and NEMO hydrodynamic data.")
    print("It allows you to setup, run and analyse connectivity data for any marine species and region of your choice.")
    print("See the User Manual XXX for more information and guidance.")
    print("You can use the package in three different ways:")
    print("1. Interactive: Guided setup of the entire process. Recommended for first time users.")
    print("2. Automatic: Run the entire process automatically using initial setup parameters.")
    print("3. Manual: Access individual functions to create a custom workflow.")
    print("\nPlease select an option (1, 2, or 3):")
def choose_mode():
    choice = input("Enter your choice (1 for Interactive, 2 for Automatic, 3 for Manual): ")

    if choice == '1':
        print("Loading interactive mode...")
        from . import interactive_functions
        return interactive_functions
    elif choice == '2':
        print("Loading automatic mode...")
        from . import automatic_functions
        return automatic_functions
    elif choice == '3':
        print("Loading manual mode...")
        from . import manual_functions
        return manual_functions
    else:
        print("Invalid choice. Please try again.")
        return choose_mode()  # Recursively ask for correct input

def import_functions_to_global(module):
    """
    Import all functions from the selected module into the global namespace.
    """
    for name in dir(module):
        if not name.startswith("_"):  # Ignore private/internal functions
            globals()[name] = getattr(module, name)

# Load necessary modules globally
load_modules()

# Show the welcome message and select mode
welcome_message()
selected_module = choose_mode()

# Import all functions from the selected module into the global namespace
import_functions_to_global(selected_module)