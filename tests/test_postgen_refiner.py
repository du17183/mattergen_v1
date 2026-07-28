import numpy as np
from ase import Atoms

from research.postgen_fastgate.refiner_eval import (
    POSITION_RADIUS_ANGSTROM,
    position_proposal,
)


def test_position_proposal_preserves_atomic_species_and_cell() -> None:
    atoms = Atoms(
        numbers=[1, 8],
        scaled_positions=[[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    forces = np.array([[10.0, 0.0, 0.0], [0.0, -10.0, 0.0]])
    proposed = position_proposal(atoms, forces, 1.0)
    assert np.array_equal(proposed.numbers, atoms.numbers)
    assert np.array_equal(proposed.cell.array, atoms.cell.array)
    displacement = proposed.positions - atoms.positions
    assert np.all(np.linalg.norm(displacement, axis=1) <= POSITION_RADIUS_ANGSTROM + 1.0e-12)


def test_force_vector_step_is_rotation_equivariant() -> None:
    atoms = Atoms(
        numbers=[6],
        positions=[[1.0, 1.0, 1.0]],
        cell=np.eye(3) * 6.0,
        pbc=True,
    )
    force = np.array([[0.4, -0.2, 0.1]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    left = position_proposal(atoms, force, 1.0).positions - atoms.positions
    rotated_atoms = atoms.copy()
    rotated_atoms.set_cell(atoms.cell.array @ rotation.T)
    rotated_atoms.positions[:] = atoms.positions @ rotation.T
    right = (
        position_proposal(rotated_atoms, force @ rotation.T, 1.0).positions
        - rotated_atoms.positions
    )
    assert np.allclose(right, left @ rotation.T)
