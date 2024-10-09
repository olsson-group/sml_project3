Getting the environments for this weeks exercises up and running is a little challenlenging so we will be installing it using the install.sh script. Make sure to have conda installed in your environment. 

Start by cloning the repository

```bash
git clone https://github.com/olsson-group/sml_project3.git
```

The install script will create an conda environment located in the sml_project3 folder. The environment is called sml3_env. 
Here we will install jupyter notebook, nglview (needed for visualizing molecules) torch packages and openff and openmm, neeeded for calculating energies.

```bash
cd sml_project3
source ./install.sh
```

After the installation is done, activate the environment and start jupyter notebook

```bash
conda activate sml3_env
jupyter notebook
```

and solve the exercises. 

Good luck!

