import mdtraj as md
import numpy as np
from mdtraj import element

ELEMENTS = {
    1: element.hydrogen,
    6: element.carbon,
}


def get_topology(atom_numbers):
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("RES", chain)

    for i, atom_number in enumerate(atom_numbers):
        e = ELEMENTS[int(atom_number)]
        name = f"{e}{i}"
        topology.add_atom(name, e, residue)

    return topology


class TorsionEvaluator:
    def __init__(self):
        self.dihedral_atoms = [[1, 2, 3, 4]]
        self.topology = get_topology([6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def evaluate(self, traj):
        assert traj.shape[1] == 15, "Trajectory must have shape (n_frames, 15, 3)"
        assert traj.shape[2] == 3, "Trajectory must have shape (n_frames, 15, 3)"

        traj = md.Trajectory(traj, self.topology)
        dihedrals = md.compute_dihedrals(traj, [[1, 2, 3, 4]])

        return dihedrals


if __name__ == "__main__":
    evaluator = TorsionEvaluator()
    traj = np.random.rand(10, 15, 3)
    dihedrals = evaluator.evaluate(traj)
    print(dihedrals)
