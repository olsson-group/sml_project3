conda create --prefix ./sml3_env python=3.10 --yes
conda activate ./sml3_env || { echo "Failed to activate environment"; exit 1; }

if [[ "$CONDA_PREFIX" != "$(pwd)/sml3_env" ]]; then
    echo "Environment not active. Exiting to avoid installing packages in global environment."
else
    echo "Environment is active. Proceeding with installations."
    pip install jupyter notebook
    pip install nglview
    pip install mdtraj
    pip install torch
    pip install torch_geometric
    pip install matplotlib
    
    pip cache purge
    pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python3 -c "import torch; print(torch.__version__)")+cu$(python3 -c "import torch; print(torch.version.cuda.split('.')[0]+torch.version.cuda.split('.')[1])")/torch_scatter.html
    pip install torch-cluster -f https://data.pyg.org/whl/torch-$(python3 -c "import torch; print(torch.__version__)")+cu$(python3 -c "import torch; print(torch.version.cuda.split('.')[0]+torch.version.cuda.split('.')[1])")/torch_cluster.html
    
    conda install -c conda-forge mdtraj --yes
fi

