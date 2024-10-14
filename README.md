# Community Connectivity Package

Welcome to the **Community Connectivity Package**! This Python package facilitates multi-species dispersal models using **Ocean Parcels** and **NEMO hydrodynamic data**. It allows users to set up, run, and analyze connectivity data for marine species in various regions.

## Installation Guide for Users

### Prerequisites:
To get started, you'll need to install **Conda** on your system. Follow the steps below:

### Step 1: Install Conda (Miniconda)
1. **Download and install Miniconda** (a minimal Conda installer):
   - **Windows**: [Download Miniconda for Windows](https://docs.conda.io/en/latest/miniconda.html)
   - **macOS**: [Download Miniconda for macOS](https://docs.conda.io/en/latest/miniconda.html)
   - **Linux**: [Download Miniconda for Linux](https://docs.conda.io/en/latest/miniconda.html)

2. **Install Conda** by running the installer for your operating system.

### Step 2: Create and Set Up the Conda Environment
Once Conda is installed, follow these steps to set up the environment for the **Community Connectivity Package**:

1. **Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux)**.

2. **Run the following command** to create a Conda environment and install all necessary dependencies:

bash
conda activate base  # Ensure you're in the base environment first
conda env create --name community_connectivity -f parcels.yml

This command will:
Create a new Conda environment with the name community_connectivity_env.
Install Python and all the required dependencies from the environment.yml file.
Activate the Conda environment:

bash
conda activate community_connectivity

Launch Jupyter Notebook:
Now that your environment is set up, you can launch Jupyter Notebook by running:

bash
jupyter notebook

This will open Jupyter in your web browser.

Step 3: Install the Community Connectivity Package
In your Jupyter Notebook, install the Community Connectivity Package by running the following command:

python
Copy code
!pip install -e .
install community_connectivity()
This will install the Community Connectivity Package in editable mode. Now, you can run and use all the functions provided by the package!

Step 4: Run the Package
Once everything is installed, you can start working with the package. For example, to choose the mode of operation, run the following in a new Jupyter cell:

python
Copy code
from community_connectivity import choose_mode
choose_mode()
Troubleshooting
If you face any issues during installation, make sure you have the latest version of Conda and that the environment is activated correctly. You can always check which environment you are in by running:

bash
Copy code
conda info --envs
If you see community_connectivity_env listed, then the environment is set up correctly.

Contact
If you encounter any issues or have any questions, feel free to contact us at [laura.bachmaier@plymouth.ac.uk].