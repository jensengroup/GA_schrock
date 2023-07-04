"""Written by Jan H. and modified by Magnus Strandgaard 2023.

Jensen 2018.
Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
"""

import copy
import random

import numpy as np
from rdkit import Chem

import crossover as co
import mutate as mu
from scoring.make_structures import create_prim_amine
from utils.classes import Generation, Individual


def read_file(file_name):
    """Read smiles from file and return mol list."""
    mol_list = []
    with open(file_name, "r") as file:
        for smiles in file:
            mol_list.append(Chem.MolFromSmiles(smiles))

    return mol_list


def make_initial_population(population_size, file_name):
    """Create starting population from csv file.

    Args:
        population_size (int): How many molecules in starting population
        file_name (str): Name of csv til to load molecules from

    Returns:
        initial_population(Generation(class)): Class containing all molecules
    """

    # Get list of moles from csv file and initialize Generation class.
    mol_list = read_file(file_name)
    initial_population = Generation(generation_num=0)

    for _ in range(population_size):

        # Randomly choose mol until we find something with any amines
        candidate_match = None
        while not candidate_match:
            mol = random.choice(mol_list)

            # Match amines, not bound to amines in rings or other amines
            candidate_match = mol.GetSubstructMatches(
                Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(*n);!$(*N)]")
            )

        # Check for prim amine to cut on
        match = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2;!$(*n);!$(*N)]"))
        # If not primary amines, create new ligand from secondary or teriary.
        if not match:
            print(f"There are no primary amines to cut so creating new")
            ligand, cut_idx = create_prim_amine(mol)

            # If we cannot split, simply add methyl as ligand (instead of discarding)
            if not cut_idx:
                ligand = Chem.MolFromSmiles("CN")
                cut_idx = [[1]]
            initial_population.molecules.append(
                Individual(ligand, cut_idx=cut_idx[0][0], original_mol=mol)
            )
        else:
            initial_population.molecules.append(
                Individual(mol, cut_idx=random.choice(match)[0], original_mol=mol)
            )
    # Assign idx to molecules to track origin
    initial_population.generation_num = 0
    initial_population.assign_idx()
    return initial_population


def make_initial_population_debug(population_size):
    """Function that runs localy and creates a small population for debugging.

    Args:
        population_size (int): How many molecules in starting population

    Returns:
        initial_population(Generation(class)): Class containing all molecules
    """
    initial_population = Generation(generation_num=0)

    # Smiles with primary amines and corresponding cut idx
    smiles = ["CCN", "NC1CCC1", "CCN", "CCN"]
    idx = [2, 0, 2, 2]

    for i in range(population_size):

        ligand = Chem.MolFromSmiles(smiles[i])
        cut_idx = [[idx[i]]]
        initial_population.molecules.append(Individual(ligand, cut_idx=cut_idx[0][0]))
    initial_population.generation_num = 0
    initial_population.assign_idx()
    return initial_population


def make_mating_pool(population, mating_pool_size):
    """Select candidates from population based on fitness(score)

    Args:
        population Generation(class): The generation object
        mating_pool_size (int): The size of the mating pool

    Returns:
        mating_pool List(Individual): List of Individual objects
    """

    fitness = population.get("normalized_fitness")
    mating_pool = []
    for _ in range(mating_pool_size):
        mating_pool.append(
            copy.deepcopy(np.random.choice(population.molecules, p=fitness))
        )

    return mating_pool


def reproduce(mating_pool, population_size, mutation_rate, molecule_filter):
    """Perform crossover operating on the molecules in the mating pool.

    Args:
        mating_pool (List(Individual)): List containing ind objects
        population_size (int): Size of whole population
        mutation_rate (float): Probability of mutation
        molecule_filter List(Chem.rdchem.Mol): List of smart pattern mol objects
            that ensure that toxic and other unwanted molecules are not evolved

    Returns:
        Generation(class): The object holding the new population
    """
    new_population = []
    # Run mutation and crossover until we have N = population_size
    while len(new_population) < population_size:
        if random.random() > mutation_rate:
            parent_A = copy.deepcopy(random.choice(mating_pool))
            parent_B = copy.deepcopy(random.choice(mating_pool))
            new_child = co.crossover(
                parent_A.rdkit_mol, parent_B.rdkit_mol, molecule_filter
            )
            if new_child:
                new_child = Individual(rdkit_mol=new_child)
                new_population.append(new_child)
        else:
            parent = copy.deepcopy(random.choice(mating_pool))
            mutated_child, mutated = mu.mutate(parent.rdkit_mol, 1, molecule_filter)
            if mutated_child:
                new_population.append(Individual(rdkit_mol=mutated_child))
    return Generation(molecules=new_population)


def sanitize(molecules, population_size):
    """Create a new population from the proposed molecules.

    If any molecules from newly scored molecules exists in population,
    we only select one. Finaly the prune class method is called to
    return only the top scoring molecules.

    Args:
        molecules List(Individual): List of molecules to operate on.
            Contains newly scored molecules and the current best molecules.
        population_size (int): How many molecules allowed in population.

    Returns:
    """

    # Dont select duplicates
    smiles_list = []
    new_population = Generation()
    for individual in molecules:
        copy_individual = copy.deepcopy(individual)
        if copy_individual.smiles not in smiles_list:
            smiles_list.append(copy_individual.smiles)
            new_population.molecules.append(copy_individual)

    # Sort by score and take the top scoring molecules.
    new_population.sort_by_score_and_prune(population_size)

    return new_population
