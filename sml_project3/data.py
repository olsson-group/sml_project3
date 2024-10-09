import os

import mdtraj as md
import numpy as np
import torch
import torch_geometric as geom

PENTENE_ATOMS = [6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


class Pentene1Dataset(geom.data.Dataset):
    def __init__(self, data="data"):
        self.data = data
        trajs = []
        for f in os.listdir(data):
            if f.endswith(".pdb"):
                traj = md.load(f"{data}/{f}")

                traj.center_coordinates()
                trajs.append(traj.xyz)

        self.atoms = torch.tensor(PENTENE_ATOMS)
        self.data = torch.tensor(np.concatenate(trajs, axis=0).tolist())
        self.edge_index = torch.tensor(
            [(i, j) for i in range(15) for j in range(15) if i != j]
        ).T

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = geom.data.Data(
            z=self.atoms,
            pos=self.data[idx],
            edge_index=self.edge_index,
            atom_idx=torch.arange(15),
        )
        return data
