"""Module containing classes used in the GA and conformer searches."""

import copy
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem
from tabulate import tabulate

from sa.neutralize import read_neutralizers
from sa.sascorer import sa_target_score_clipped
from scoring.make_structures import atom_remover, create_prim_amine, single_atom_remover
from utils.gaussians import number_of_rotatable_bonds_target_clipped


@dataclass
class Individual:
    """Dataclass for storing data for each molecule in the GA.

    The central objects of the GA. The moles themselves plus various
    attributes and debugging fields are set.

    Attributes:
        rdkit_mol: The rdkit mol object
        original_mol: The mol object at the start of a generation.
        rdkit_mol_sa: Mol object where the primary amine is replaced with a hydrogen.
         Used for the SA score.
        optimized_mol1: The mol object of Schrock core + moiety for the first structure in the scoring
        function. The mol contains the optimized geometries.
        optimized_mol2: The mol object of Schrock core + moiety for the second structure in the scoring
        function. The mol contains the optimized geometries.
        cut_idx: The index of the primary amine that denotes the attachment point.
        idx: The generation idx of the molecule.
        smiles: SMILES representation of molecule.
        smiles_sa: SMILES representation of the molecule with primary amine replaced with
        hydrogen for SA score.
        score: GA score for the molecule.
        normalized_fitness: Normalized score value for the current population.
        energy: Reaction energy for the scoring step.
        sa_score: Synthetic accessibility score.
    """

    rdkit_mol: Chem.rdchem.Mol = field(repr=False, compare=False)
    original_mol: Chem.rdchem.Mol = field(
        default_factory=Chem.rdchem.Mol, repr=False, compare=False
    )
    rdkit_mol_sa: Chem.rdchem.Mol = field(
        default_factory=Chem.rdchem.Mol, repr=False, compare=False
    )
    optimized_mol1: Chem.rdchem.Mol = field(
        default_factory=Chem.rdchem.Mol, repr=False, compare=False
    )
    optimized_mol2: Chem.rdchem.Mol = field(
        default_factory=Chem.rdchem.Mol, repr=False, compare=False
    )
    cut_idx: int = field(default=None, repr=False, compare=False)
    idx: tuple = field(default=(None, None), repr=False, compare=False)
    smiles: str = field(init=False, compare=True, repr=True)
    smiles_sa: str = field(init=False, compare=True, repr=False)
    score: float = field(default=None, repr=False, compare=False)
    normalized_fitness: float = field(default=None, repr=False, compare=False)
    energy: float = field(default=None, repr=False, compare=False)
    sa_score: float = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.smiles = Chem.MolToSmiles(self.rdkit_mol)

    def list_of_props(self):
        return [
            self.idx,
            self.normalized_fitness,
            self.score,
            self.energy,
            self.sa_score,
            self.smiles,
        ]

    def get(self, prop):
        """Get property from individual."""
        prop = getattr(self, prop)
        return prop

    def save(self, directory="."):
        """Dump ind object into file."""
        filename = os.path.join(directory, f"ind.pkl")
        with open(filename, "ab+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


@dataclass(order=True)
class Generation:
    """Dataclass holding the Individuals in each generation.

    Contains functionality to get and set props from Individuals and
    display various scoring results
    """

    molecules: List[Individual] = field(repr=True, default_factory=list)
    new_molecules: List[Individual] = field(repr=False, default_factory=list)
    generation_num: int = field(init=True, default=None)
    size: int = field(default=None, init=True, repr=True)

    def __post_init__(self):
        self.size = len(self.molecules)

    def __repr__(self):
        return (
            f"" f"(generation_num={self.generation_num!r}, molecules_size={self.size})"
        )

    def assign_idx(self):
        """Set idx on each molecule."""
        for i, molecule in enumerate(self.molecules):
            setattr(molecule, "idx", (self.generation_num, i))
        self.size = len(self.molecules)

    def save(self, directory=None, name="GA.pkl"):
        """Save instance to file for later retrieval."""
        filename = os.path.join(directory, name)
        with open(filename, "ab+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def get(self, prop):
        """Get property from molecules."""
        properties = []
        for molecule in self.molecules:
            properties.append(getattr(molecule, prop))
        return properties

    def setprop(self, prop, list_of_values):
        """Set property for molecules."""
        for molecule, value in zip(self.molecules, list_of_values):
            setattr(molecule, prop, value)

    def appendprop(self, prop, list_of_values):
        for molecule, value in zip(self.molecules, list_of_values):
            if value:
                getattr(molecule, prop).append(value)

    def sortby(self, prop, reverse=True):
        """Sort molecule based on score."""
        if reverse:
            self.molecules.sort(
                key=lambda x: float("inf") if np.isnan(x.score) else x.score,
                reverse=reverse,
            )
        else:
            self.molecules.sort(
                key=lambda x: float("inf") if np.isnan(x.score) else x.score,
                reverse=reverse,
            )

    def handle_results(self, results):
        """Extract the scoring results and set the properties on the Individual
        objects."""
        optimized_mol1 = [res[0] for res in results]
        optimized_mol2 = [res[1] for res in results]
        en_dicts = [res[2] for res in results]
        self.setprop("energy_dict", en_dicts)

        # Extract scores and set on the Individals
        self.setprop("score", [en["score"] for en in en_dicts])
        self.setprop("pre_score", [en["score"] for en in en_dicts])
        self.setprop("energy", [en["score"] for en in en_dicts])

        for mol, opt1, opt2 in zip(self.molecules, optimized_mol1, optimized_mol2):
            mol.optimized_mol1 = opt1
            mol.optimized_mol2 = opt2

    def reweigh_rotatable_bonds(self, nrb_target=4, nrb_standard_deviation=2):
        """Scale the current scores by the number of rotational bonds.

        Args:
            nrb_target: Limit for number of rotational bonds.
            nrb_standard_deviation: STD defines the width of the gaussian above the limit nrb_target.
        """
        number_of_rotatable_target_scores = [
            number_of_rotatable_bonds_target_clipped(
                p.rdkit_mol, nrb_target, nrb_standard_deviation
            )
            for p in self.molecules
        ]

        new_scores = [
            score * scale
            for score, scale in zip(
                self.get("score"), number_of_rotatable_target_scores
            )
        ]
        self.setprop("score", new_scores)

    def sort_by_score_and_prune(self, population_size):
        """Sort by score and take the best scoring molecules."""
        self.sortby("score", reverse=False)
        self.molecules = self.molecules[:population_size]
        self.size = len(self.molecules)

    def get_text(self, population="molecules", pass_text=None):
        """Print nice table of population attributes."""
        table = []
        if population == "molecules":
            population = self.molecules
        elif population == "new_molecules":
            population = self.new_molecules
        for individual in population:
            table.append(individual.list_of_props())
        txt = tabulate(
            table,
            headers=[
                "idx",
                "normalized_fitness",
                "score",
                "energy",
                "sa_score",
                "smiles",
            ],
        )
        return txt

    def print_fails(self):
        """Log how many calcs in population failed."""
        nO_NaN = 0
        nO_9999 = 0
        for ind in self.new_molecules:
            tmp = ind.energy
            if np.isnan(tmp):
                nO_NaN += 1
            elif tmp > 5000:
                nO_9999 += 1
        table = [[nO_NaN, nO_9999, nO_NaN + nO_9999]]
        txt = tabulate(
            table,
            headers=["Number of NaNs", "Number of high energies", "Total"],
        )
        return txt

    def gen2pd(
        self,
        columns=["cut_idx", "score", "energy", "sa_score", "smiles"],
    ):
        """Get dataframe of population."""
        df = pd.DataFrame(
            list(map(list, zip(*[self.get(prop) for prop in columns]))),
            index=pd.MultiIndex.from_tuples(
                self.get("idx"), names=("generation", "individual")
            ),
        )
        df.columns = columns
        return df

    def gen2pd_dft(self):
        columns = [
            "smiles",
            "idx",
            "cut_idx",
            "score",
            "energy",
            "dft_singlepoint_conf",
            "min_confs",
        ]
        """Get dataframe of population."""
        df = pd.DataFrame(list(map(list, zip(*[self.get(prop) for prop in columns]))))
        df.columns = columns
        return df

    def update_property_cache(self):
        """Update rdkit data to prevent errors."""
        for mol in self.molecules:
            # Done to prevent ringinfo error
            Chem.GetSymmSSSR(mol.rdkit_mol)
            mol.rdkit_mol.UpdatePropertyCache()

    def modify_population(self, supress_amines=False):
        """Molecule mol modifier function. Preps molecules in population for
        scoring. Ensures that there is one primary amine attachment point.

        supress_amines: Decides whether primary amines other than the
        attachment point are changed to hydrogen.
        """
        # Loop over molecules in popualtion
        for mol in self.molecules:
            # Check for primary amine
            match = mol.rdkit_mol.GetSubstructMatches(
                Chem.MolFromSmarts("[NX3;H2;!$(*n);!$(*N)]")
            )
            # Set current mol for future debugging
            mol.original_mol = mol.rdkit_mol

            # Create primary amine if it doesnt have one.
            if not match:
                try:
                    output_ligand, cut_idx = create_prim_amine(mol.rdkit_mol)

                    # Handle if None is returned
                    if not (output_ligand or cut_idx):
                        output_ligand = Chem.MolFromSmiles("CCCCCN")
                        cut_idx = [[1]]
                except Exception as e:
                    print("Could not create primary amine, setting methyl as ligand")
                    output_ligand = Chem.MolFromSmiles("CN")
                    cut_idx = [[1]]

                # rdkit hack to ensure smiles look ok
                mol.rdkit_mol = output_ligand
                mol.cut_idx = cut_idx[0][0]
                mol.smiles = Chem.MolToSmiles(output_ligand)

            else:
                cut_idx = random.choice(match)
                mol.cut_idx = cut_idx[0]

                # Remove additional primary amine groups to prevent XTB exploit
                if supress_amines:
                    # Check for N-N bound amines
                    nn_match = mol.rdkit_mol.GetSubstructMatches(
                        Chem.MolFromSmarts("[NX3;H2;$(*N),$(*n)]")
                    )

                    # Enable NH2 amine supressor if there are multiple
                    # primary amines
                    if len(match) > 1:
                        # Substructure match the NH3
                        prim_match = Chem.MolFromSmarts("[NX3;H2]")

                        # Remove the primary amines
                        ms = [
                            x for x in atom_remover(mol.rdkit_mol, pattern=prim_match)
                        ]
                        removed_mol = random.choice(ms)
                        prim_amine_index = removed_mol.GetSubstructMatches(
                            Chem.MolFromSmarts("[NX3;H2]")
                        )
                        mol.rdkit_mol = removed_mol
                        mol.cut_idx = prim_amine_index[0][0]
                        mol.smiles = Chem.MolToSmiles(removed_mol)

                    elif nn_match:
                        # Replace tricky primary amines in the frag:
                        prim_match = Chem.MolFromSmarts("[NX3;H2;$(*N),$(*n)]")

                        rm = Chem.ReplaceSubstructs(
                            mol.rdkit_mol,
                            prim_match,
                            Chem.MolFromSmiles("[H]"),
                            replaceAll=True,
                        )[0]
                        rm = Chem.RemoveHs(rm)
                        prim_amine_index = rm.GetSubstructMatches(
                            Chem.MolFromSmarts("[NX3;H2]")
                        )
                        mol.rdkit_mol = rm
                        mol.cut_idx = prim_amine_index[0][0]
                        mol.smiles = Chem.MolToSmiles(rm)

    ### SA functionality
    def sa_prep(self):
        for mol in self.molecules:
            prim_match = Chem.MolFromSmarts("[NX3;H2]")
            # Remove the cut idx amine to prevent it hogging the SA score
            removed_mol = single_atom_remover(mol.rdkit_mol, mol.cut_idx)
            mol.rdkit_mol_sa = removed_mol
            mol.smiles_sa = Chem.MolToSmiles(removed_mol)

            _neutralize_reactions = read_neutralizers()

        neutral_molecules = []
        for ind in self.molecules:
            c_mol = ind.rdkit_mol_sa
            mol = copy.deepcopy(c_mol)
            mol.UpdatePropertyCache()
            Chem.rdmolops.FastFindRings(mol)
            assert mol is not None
            for reactant_mol, product_mol in _neutralize_reactions:
                while mol.HasSubstructMatch(reactant_mol):
                    rms = Chem.ReplaceSubstructs(mol, reactant_mol, product_mol)
                    if rms[0] is not None:
                        mol = rms[0]
            mol.UpdatePropertyCache()
            Chem.rdmolops.FastFindRings(mol)
            ind.neutral_rdkit_mol = mol

    def get_sa(self):
        """Get the SA score of the population."""

        # Neutralize and prep molecules
        self.sa_prep()

        # Get the scores
        sa_scores = [
            sa_target_score_clipped(ind.neutral_rdkit_mol) for ind in self.molecules
        ]
        # Set the scores
        self.set_sa(sa_scores)

    def calculate_normalized_fitness(self):
        """Normalize the scores to get probabilities for mating selection."""

        # onvert to high and low scores.
        scores = self.get("score")
        scores = [-s for s in scores]

        min_score = np.nanmin(scores)
        shifted_scores = [
            0 if np.isnan(score) else score - min_score for score in scores
        ]
        sum_scores = sum(shifted_scores)
        if sum_scores == 0:
            print(
                "WARNING: Shifted scores are zero. Normalized fitness is therefore dividing with "
                "zero, could be because the population only contains one individual"
            )

        for individual, shifted_score in zip(self.molecules, shifted_scores):
            individual.normalized_fitness = shifted_score / sum_scores

    def set_sa(self, sa_scores):
        """Set sa score."""
        for individual, sa_score in zip(self.molecules, sa_scores):
            individual.sa_score = sa_score
            # Scale the score with the sa_score (which is max 1)
            individual.score = sa_score * individual.pre_score


@dataclass(order=True)
class Conformers:
    """Dataclass holding the molecules in the conformer screening.

    Contains functionality to get and set props from Individuals and
    display various scoring results
    """

    molecules: List[Individual] = field(repr=True, default_factory=list)

    @property
    def size(self):
        return len(self.molecules)

    def __repr__(self):
        return f"molecules_size={self.size})"

    def save(self, directory=None, name="Conformers.pkl"):
        """Save instance to file for later retrieval."""
        filename = os.path.join(directory, name)
        with open(filename, "ab+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def get(self, prop):
        """Get property from molecules."""
        properties = []
        for molecule in self.molecules:
            properties.append(getattr(molecule, prop))
        return properties

    def setprop(self, prop, list_of_values):
        """Set property for molecules."""
        for molecule, value in zip(self.molecules, list_of_values):
            setattr(molecule, prop, value)

    def appendprop(self, prop, list_of_values):
        for molecule, value in zip(self.molecules, list_of_values):
            if value:
                getattr(molecule, prop).append(value)

    def sortby(self, prop, reverse=False):
        """Sort molecule based on score."""
        if reverse:
            self.molecules.sort(
                key=lambda x: float("inf") if np.isnan(x.get(prop)) else x.get(prop),
                reverse=reverse,
            )
        else:
            self.molecules.sort(
                key=lambda x: float("inf") if np.isnan(x.get(prop)) else x.get(prop),
                reverse=reverse,
            )

    def set_results(self, results):
        """Extract the scoring results and set the Individual properties."""
        energies = [res[0] for res in results]
        geometries = [res[1] for res in results]
        geometries2 = [res[2] for res in results]
        min_conf = [res[3] for res in results]

        self.setprop("energy", energies)
        self.setprop("pre_score", energies)
        self.setprop("structure", geometries)
        self.setprop("structure2", geometries2)
        self.setprop("min_conf", min_conf)
        self.setprop("score", energies)

    def handle_results(self, results):
        """Extract the scoring results and set the properties on the Individual
        objects."""
        optimized_mol1 = [res[0] for res in results]
        optimized_mol2 = [res[1] for res in results]
        en_dicts = [res[2] for res in results]
        self.setprop("energy_dict", en_dicts)

        # Extract scores and set on the Individals
        self.setprop("score", [en["score"] for en in en_dicts])
        self.setprop("pre_score", [en["score"] for en in en_dicts])
        self.setprop("energy", [en["score"] for en in en_dicts])

        for mol, opt1, opt2 in zip(self.molecules, optimized_mol1, optimized_mol2):
            mol.optimized_mol1 = opt1
            mol.optimized_mol2 = opt2

    def conf2pd_dft(self):
        """Get dataframe of population."""
        columns = (
            [
                "smiles",
                "idx",
                "cut_idx",
                "score",
                "energy",
                "dft_singlepoint_conf",
                "final_dft_opt",
                "scoring_function",
            ],
        )
        df = pd.DataFrame(list(map(list, zip(*[self.get(prop) for prop in columns]))))
        df.columns = columns
        return df
