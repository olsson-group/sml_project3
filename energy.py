import openmm
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule, Topology
from openmm import unit


def setup_butane_openmm_molecule_system():
    force_field = ForceField("openff-2.0.0.offxml")
    molecule = Molecule.from_smiles("C=CCCC")
    molecule.generate_conformers(n_conformers=1)
    molecule.assign_partial_charges("mmff94", use_conformers=molecule.conformers)

    topology = Topology.from_molecules([molecule])
    interchange = Interchange.from_smirnoff(force_field, topology)

    openmm_topology = interchange.to_openmm_topology()
    openmm_system = interchange.to_openmm()
    openmm_positions = interchange.positions.to_openmm()
    return openmm_topology, openmm_system, openmm_positions


def evaluate_energies(openmm_topology, openmm_system, configurations):
    time_step = 0.5 * unit.femtoseconds
    temperature = 400 * unit.kelvin
    friction = 1 / unit.picosecond

    integrator = openmm.LangevinIntegrator(temperature, friction, time_step)

    simulation = openmm.app.Simulation(openmm_topology, openmm_system, integrator)
    energies = []
    for conf in configurations:
        simulation.context.setPositions(conf)
        state = simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        energies.append(energy)

    return energies


class EnergyEvaluator:
    def __init__(self):
        self.openmm_topology, self.openmm_system, self.openmm_positions = (
            setup_butane_openmm_molecule_system()
        )

    def evaluate(self, configurations):
        return evaluate_energies(
            self.openmm_topology, self.openmm_system, configurations
        )
