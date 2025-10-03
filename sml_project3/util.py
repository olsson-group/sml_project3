from datetime import datetime

import mdtraj as md
import nglview
import torch_geometric as geom
from torch_scatter import scatter

PENTENE_ATOMS = [6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def center_coordinates(x):
    com = x.mean(dim=1, keepdim=True).repeat(1, 15, 1)
    return x - com


def center_batch(batch):
    com = scatter(batch.x, batch.batch, dim=0, reduce="mean")
    batch.x -= com[batch.batch]
    return batch


class Timer:
    def __init__(self):
        self.start = datetime.now()

    def __str__(self):
        now = datetime.now()
        time_passed = now - self.start
        return str(time_passed).split(".")[0]


def get_example_batch(dataset, batch_size):
    example_batch = next(iter(geom.loader.DataLoader(dataset, batch_size=batch_size)))
    return example_batch


ELEMENTS = {
    1: md.element.hydrogen,
    6: md.element.carbon,
}


def get_pentene_mdtraj(traj):
    topology = get_topology(PENTENE_ATOMS)
    return md.Trajectory(traj, topology)


def nglview_pentene(traj):
    traj = get_pentene_mdtraj(traj)
    view = nglview.show_mdtraj(traj)
    view.clear_representations()
    view.add_representation("hyperball")
    return view


def get_topology(atom_numbers):
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("RES", chain)

    for i, atom_number in enumerate(atom_numbers):
        e = ELEMENTS[int(atom_number)]
        name = f"{e}{i}"
        topology.add_atom(name, e, residue)

    return topology
