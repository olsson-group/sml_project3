import os

import mdtraj as md
import numpy as np
import torch
import torch_geometric as geom

t1 = md.load("data/trj0.pdb")


class Pentene1Dataset(geom.data.Dataset):
    def __init__(self, data="data"):
        self.data = data
        trajs = []
        for f in os.listdir(data):
            if f.endswith(".pdb"):
                traj = md.load(f"{data}/{f}")
                trajs.append(traj.xyz)

        self.atoms = torch.tensor([atom.element.number for atom in traj.top.atoms])  # type: ignore
        self.data = torch.tensor(np.concatenate(trajs, axis=0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = geom.data.Data(
            z=self.atoms,
            pos=self.data[idx],
        )
        return data


if __name__ == "__main__":
    ds = Pentene1Dataset()
