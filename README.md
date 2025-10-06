For this project you can either choose to use Google Colab or run things locally. If you choose to use Colab you can just upload the notebook and follow the installation instructions from there. The second option is to run things locally. In that case, please follow the installation instructions below.

Getting the environments for this weeks exercises up and running is a little challenenging so we will be installing it using the install.sh script. Make sure to have conda installed in your environment. 

Start by cloning the repository

```bash
git clone https://github.com/olsson-group/sml_project3.git
```

The install script will create an conda environment located in the sml_project3 folder. The environment is called sml3_env. Here we will install jupyter notebook, nglview (needed for visualizing molecules) and torch packages. Run the commands below.

```bash
cd sml_project3
source ./install.sh
```

After the installation is done, place the notebook you downloaded from the assignment page (`sml_project3.ipynb`) at the top level of the directory. Then activate the environment and start jupyter notebook

```bash
conda activate sml3_env
jupyter notebook
```

and solve the exercises in 

```bash
sml_project3.ipynb
```

Good luck!
