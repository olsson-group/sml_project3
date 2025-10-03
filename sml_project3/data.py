import os

import mdtraj as md
import numpy as np
import torch
import torch_geometric as geom

PENTENE_ATOMS = [6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


class Pentene1Dataset(geom.data.Dataset):
    def __init__(self, data=None):
        if data is None:
            data = "data"
            trajs = []
            for f in os.listdir(data):
                if f.endswith(".pdb"):
                    traj = md.load(f"{data}/{f}")

                    traj.center_coordinates()
                    trajs.append(traj.xyz)
            self.data = torch.tensor(np.concatenate(trajs, axis=0).tolist())

        else:  # if data is already provided
            self.data = torch.tensor(data)

        self.atoms = torch.tensor(PENTENE_ATOMS)
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


class BaseDistributionDataset(geom.data.Dataset):
    def __init__(self, n_samples, basedistribution):
        self.data = basedistribution.sample(n_samples).view(-1, 15, 3)
        assert self.data.shape == (n_samples, 15, 3)

        self.atoms = torch.tensor(PENTENE_ATOMS)
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
