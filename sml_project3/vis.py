import mdtraj as md
import torch

from sml_project3.data import PENTENE_ATOMS

ELEMENTS = {
    1: md.element.hydrogen,
    6: md.element.carbon,
}


def get_pentene_mdtraj(traj):
    topology = get_topology(PENTENE_ATOMS)
    return md.Trajectory(traj, topology)


def nglview_pentene(traj):
    traj = get_pentene_mdtraj(traj)
    view = md.show_mdtraj(traj)
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
