import json
import re
from argparse import ArgumentParser

import numpy as np
from openff.toolkit.topology import Molecule, Topology
from openmm import Context, VerletIntegrator
from openmm.app import ForceField
from openmmforcefields.generators import GAFFTemplateGenerator
from simtk import unit
from simtk.openmm import HarmonicBondForce, PeriodicTorsionForce
from tqdm import tqdm

from tito import mlops


def calculate_conformation_energies(
    rd_mol, partial_charges, traj, forcefield_path="amber/protein.ff14SB.xml"
):
    off_mol = Molecule.from_rdkit(rd_mol)
    off_mol.partial_charges = unit.Quantity(
        value=np.array(partial_charges), unit=unit.elementary_charge
    )
    gaff = GAFFTemplateGenerator(molecules=off_mol)
    topology = Topology.from_molecules(off_mol).to_openmm()
    forcefield = ForceField(forcefield_path)
    forcefield.registerTemplateGenerator(gaff.generator)
    system = forcefield.createSystem(topology)

    for i, force in enumerate(system.getForces()):
        if isinstance(force, HarmonicBondForce):
            force.setForceGroup(1)
        elif isinstance(force, PeriodicTorsionForce):
            force.setForceGroup(2)

    integrator = VerletIntegrator(
        1.0 * unit.femtoseconds
    )  # Minimal integrator for energy calculation
    context = Context(system, integrator)

    bond_energies = []
    torsional_energies = []
    total_energies = []

    for conformation in traj:
        context.setPositions(conformation)

        state_total = context.getState(getEnergy=True)
        total_energy = state_total.getPotentialEnergy().value_in_unit(
            unit.kilojoule / unit.mole
        )
        total_energies.append(total_energy)

        state_bonds = context.getState(getEnergy=True, groups={1 << 0})
        bond_energy = state_bonds.getPotentialEnergy().value_in_unit(
            unit.kilojoule / unit.mole
        )
        bond_energies.append(bond_energy)

        state_torsions = context.getState(getEnergy=True, groups={1 << 1})
        torsional_energy = state_torsions.getPotentialEnergy().value_in_unit(
            unit.kilojoule / unit.mole
        )
        torsional_energies.append(torsional_energy)

    return total_energies, bond_energies, torsional_energies


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", nargs="?", default="results/latest")
    parser.add_argument("--re", default=".*")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    main(args)
