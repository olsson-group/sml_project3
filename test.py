import warnings
from datetime import datetime

import mdtraj
import nglview
import torch
import torch_geometric as geom

from sml_project3 import data, mlops, utils
from sml_project3.model import CFM
from sml_project3.painn import Painn

#  from torchcfm.utils import plot_trajectories, sample_8gaussians, sample_moons


class CenteredNormal:
    def sample(self, batch_size):
        x = torch.randn((batch_size, 15, 3))
        x = utils.center_coordinates(x)
        x *= 0.1277

        return x.reshape(-1, 3)


class Readout(torch.nn.Module):
    def __init__(self, n_features=8, n_features_out=1):
        super().__init__()
        self.mlp = MLP(n_features, n_features, n_features_out)
        self.V = EquivariantLinear(n_features, n_features_out)
        self.n_features_out = n_features_out

    def forward(self, batch):
        gates = self.mlp(batch.invariant_node_features)
        equivariant_node_features = self.V(batch.equivariant_node_features)
        equivariant_node_features_out = multiply_first_dim(
            equivariant_node_features, gates
        )

        return equivariant_node_features_out


def multiply_first_dim(w, x):
    with warnings.catch_warnings(record=True):
        return (w.T * x.T).T


class EquivariantLinear(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super().__init__()
        self.linear = torch.nn.Linear(n_features_in, n_features_out, bias=False)

    def forward(self, x):
        return self.linear(x.swapaxes(-1, -2)).swapaxes(-1, -2)


class MLP(torch.nn.Module):
    def __init__(self, f_in, f_hidden, f_out):
        super().__init__()

        self.f_out = f_out

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(f_in, f_hidden),
            torch.nn.LayerNorm(f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_hidden),
            torch.nn.LayerNorm(f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_out),
        )

    def forward(self, x):
        return self.mlp(x)


dataset = data.Pentene1Dataset("data")
dataloader = geom.data.DataLoader(dataset, batch_size=100, shuffle=True)

basedistribution = CenteredNormal()
example_batch = utils.get_example_batch(dataset, 10)

readout = Readout()

score = Painn(n_features=8, readout=readout)

cfm = CFM(score, basedistribution)

step = 0
timer = utils.Timer()

for epoch in range(10000):
    for batch in dataloader:
        t = torch.rand(len(batch)).type_as(batch.pos)
        loss = cfm.get_loss(t, batch)
        cfm.training_step(loss)
        step += 1
        print(
            f"epoch: {epoch}, step: {step}, time passed: {timer}, loss: {loss.item():.4f}",
            end="\r",
        )

        if (step + 1) % 10 == 0:
            mlops.save(cfm, f"results/model/painn_{step}.pkl")

            traj = cfm.sample(example_batch)
            # mlops.save(traj, f"results/samples/painn_{step}.pkl")

            # nglview(traj, torch.tensor(data.PENTENE_ATOMS))
            # path = '/tmp/traj.pdb'
            traj = get_mdtraj(traj, data.PENTENE_ATOMS)

            view = nglview.show_mdtraj(traj)
            view.clear_representations()
            view.add_representation("hyperball")
            view
            break
    break
