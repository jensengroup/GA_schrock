import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds


def gaussian_modifier(score: float, target: float, sigma: float) -> float:
    """Modifies a score to a fitness to a target using a gaussian distribution
    If the score matches the target value, the function evaluates to 1. The
    width of the distribution is controlled through sigma.

    :param score: the score to evaulate fitness for
    :param target: the target value to use for fitness evaluation
    :param sigma: the width of the distribution
    :return: the fitness evaluated as a gaussian
    """
    score = np.exp(-0.5 * np.power((score - target) / sigma, 2.0))
    return score


def gaussian_modifier_clipped(score: float, target: float, sigma: float) -> float:
    """Modifies a score with a clipped gaussian modifier The clipped gaussian
    modifier is evaluated to 1 for scores below the target, but drops off as
    the gaussian modifier (towards zero) for values larger than the target. The
    width of the distribution is controlled through sigma.

    :param score: the score to evaulate fitness for
    :param target: the target value to use for fitness evaluation
    :param sigma: the width of the distribution
    :return: the fitness evaluated as a gaussian
    """

    mod_score = np.maximum(score, target)
    return gaussian_modifier(mod_score, target, sigma)


def number_of_rotatable_bonds(mol: Chem.Mol) -> int:
    """Computes the number of rotatable bonds in an RDKit molecule.

    :param mol: the molecule
    """
    return CalcNumRotatableBonds(mol)


def number_of_rotatable_bonds_target(
    mol: Chem.Mol, target: float, sigma: float
) -> float:
    n: int = number_of_rotatable_bonds(mol)
    return gaussian_modifier(n, target, sigma)


def number_of_rotatable_bonds_target_clipped(
    mol: Chem.Mol, target: float, sigma: float
) -> float:
    n: int = number_of_rotatable_bonds(mol)
    return gaussian_modifier_clipped(n, target, sigma)
