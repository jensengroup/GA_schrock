# -*- coding: utf-8 -*-
"""Scoring module handling the scoring of molecule candidates."""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

from utils.utils import energy_filter

scoring_dir = os.path.dirname(__file__)
sys.path.append(scoring_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from make_structures import (
    connect_ligand,
    create_dummy_ligand,
    embed_rdkit,
    remove_N2,
    remove_NH3,
)

from utils.utils import cd
from utils.xtb_utils import XTB_optimize_schrock

# Constants for getting score reaction energies.
# Obtained from xTB gfn1/gfn2 electronic energies.
NH3_ENERGY_gfn2 = -4.427496335658
N2_ENERGY_gfn2 = -5.766345142003
CP_RED_ENERGY_gfn2 = 0.2788559959203811
NH3_ENERGY_gfn1 = -4.834742774551
N2_ENERGY_gfn1 = -6.331044264474
CP_RED_ENERGY_gfn1 = 0.2390159933706209
GAS_ENERGIES = {
    "2": (NH3_ENERGY_gfn2, N2_ENERGY_gfn2, CP_RED_ENERGY_gfn2),
    "1": (NH3_ENERGY_gfn1, N2_ENERGY_gfn1, CP_RED_ENERGY_gfn1),
}
hartree2kcalmol = 627.51

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data")))

file = str(source / "templates/core_dummy.sdf")
core = Chem.SDMolSupplier(file, removeHs=False, sanitize=False)
"""Mol:
mol object of the Mo core with dummy atoms instead of ligands
"""

file_NH3 = str(source / "templates/core_NH3_dummy.sdf")
core_NH3 = Chem.SDMolSupplier(file_NH3, removeHs=False, sanitize=False)
"""Mol:
mol object of the Mo core with NH3 in axial position and
dummy atoms instead of ligands
"""

file_N2_NH3 = str(source / "templates/core_N2_NH3_dummy.sdf")
core_N2_NH3 = Chem.SDMolSupplier(file_N2_NH3, removeHs=False, sanitize=False)
"""Mol:
mol object of the Mo core with NH3 in axial position and
dummy atoms instead of ligands
"""


with open(str(source / "intermediate_smiles.json"), "r", encoding="utf-8") as f:
    smi_dict = json.load(f)
"""dict:
Dictionary that contains the smiles string for each N-related intermediate
and the charge and spin for the specific intermediate
"""


def scoring_submitter(mol, scoring_args):
    """Utility function to distribute molecules to their designates scoring
    function.

    Args:
        mol (Chem.mol): The mol object to score
        scoring_args (dict): Scoring arguments.

    Returns:
    """

    scoring_args["output_dir"] = scoring_args["output_dir"] / mol.scoring_function

    if mol.scoring_function == "rdkit_embed_scoring":
        new_mol, new_mol2, en_dict = rdkit_embed_scoring(mol, scoring_args)
    elif mol.scoring_function == "rdkit_embed_scoring_NH3toN2":
        new_mol, new_mol2, en_dict = rdkit_embed_scoring_NH3toN2(mol, scoring_args)
    elif mol.scoring_function == "rdkit_embed_scoring_NH3plustoNH3":
        new_mol, new_mol2, en_dict = rdkit_embed_scoring_NH3plustoNH3(mol, scoring_args)
    else:
        print("Non valid scoring function")
        return None

    return new_mol, new_mol2, en_dict


def rdkit_embed_scoring(ligand, scoring_args):
    """Score the NH3 -> N2 binding.

    Args:
        ligand (Chem.rdchem.Mol): ligand to put on Mo core
        scoring_args (dict): dict with all relevant args for xtb and general scoring

    Returns:
    """

    # Get ligand tuple idx.
    idx = ligand.idx

    # Unpack gas energies
    NH3_ENERGY, N2_ENERGY, CP_RED_ENERGY = GAS_ENERGIES[scoring_args["method"]]

    # Get mol object of Mo core + connected ligand
    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    Mo_N2_NH3 = connect_ligand(core_N2_NH3[0], ligand_cut, NH3_flag=True, N2_flag=True)
    Mo_N2_NH3 = Chem.AddHs(Mo_N2_NH3)

    # Embed mol object
    Mo_N2_NH3_3d = embed_rdkit(
        mol=Mo_N2_NH3,
        core=core_N2_NH3[0],
        numConfs=scoring_args["n_confs"],
        pruneRmsThresh=scoring_args["RMS_thresh"],
    )

    # Go into the output directory
    with cd(scoring_args["output_dir"]):
        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_N2_NH3"
        scoring_args["charge"] = smi_dict["Mo_N2_NH3"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_N2_NH3"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_N2_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        optimized_mol1, energies = optimizer.optimize_schrock()

    # Get conformers
    confs = optimized_mol1.GetConformers()
    if len(confs) == 0:
        return None, None, {"energy1": None, "energy2": None, "score": np.nan}

    # Remove conformers that are too far from the minimum energy conf
    energies, new_mol = energy_filter(confs, energies, optimized_mol1, scoring_args)

    # Create duplicate mol object for removing conformers
    single_conf = copy.deepcopy(new_mol)

    # During GA scoring, only the last energy conformer is needed for the next step.
    # The rest of the conformers are discarded.
    if scoring_args.get("ga_scoring", False):
        # Remove higher energy conformes from mol object
        minidx = np.argmin(energies)
        discard_conf = [
            x for x in range(len(single_conf.GetConformers())) if x != minidx
        ]
        for elem in discard_conf:
            single_conf.RemoveConformer(elem)

    # Remove N2 on the full embedding
    Mo_NH3_3d = remove_N2(single_conf)
    Mo_NH3_3d = Chem.AddHs(Mo_NH3_3d)

    with cd(scoring_args["output_dir"]):
        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3"
        scoring_args["charge"] = smi_dict["Mo_NH3"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_NH3"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        optimized_mol2, energies2 = optimizer.optimize_schrock()

    confs = optimized_mol2.GetConformers()
    if len(confs) == 0:
        return None, None, {"energy1": None, "energy2": None, "score": np.nan}

    energies2, new_mol2 = energy_filter(confs, energies2, optimized_mol2, scoring_args)

    energy_diff = (energies.min() - (energies2.min() + N2_ENERGY)) * hartree2kcalmol
    print(f"Score for top scoring conformer: {energy_diff}")

    en_dict = {"energy1": energies, "energy2": energies2, "score": energy_diff}

    return new_mol, new_mol2, en_dict


def rdkit_embed_scoring_NH3toN2(ligand, scoring_args):
    """Score the NH3 -> N2 exchange.

    Args:
        ligand (Chem.rdchem.Mol): ligand to put on Mo core
        scoring_args (dict): dict with all relevant args for xtb and general scoring

    Returns:
    """

    # Get tuple idx
    idx = ligand.idx

    # Unpack gas energies
    NH3_ENERGY, N2_ENERGY, CP_RED_ENERGY = GAS_ENERGIES[scoring_args["method"]]

    # Get mol object of Mo core + connected ligand
    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    Mo_NH3 = connect_ligand(core_NH3[0], ligand_cut, NH3_flag=True)
    Mo_NH3 = Chem.AddHs(Mo_NH3)

    # Embed mol object
    Mo_NH3_3d = embed_rdkit(
        mol=Mo_NH3,
        core=core_NH3[0],
        numConfs=scoring_args["n_confs"],
        pruneRmsThresh=scoring_args["RMS_thresh"],
    )

    with cd(scoring_args["output_dir"]):
        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3"
        scoring_args["charge"] = smi_dict["Mo_NH3"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_NH3"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        optimized_mol1, energies = optimizer.optimize_schrock()

    confs = optimized_mol1.GetConformers()
    if len(confs) == 0:
        return None, None, {"energy1": None, "energy2": None, "score": np.nan}

    # Remove conformers that are too far from the minimum energy conf
    energies, new_mol = energy_filter(confs, energies, optimized_mol1, scoring_args)

    single_conf = copy.deepcopy(new_mol)

    minidx = np.argmin(energies)
    discard_conf = [x for x in range(len(single_conf.GetConformers())) if x != minidx]
    for elem in discard_conf:
        single_conf.RemoveConformer(elem)

    # Replace NH3 with N2
    Mo_N2 = Chem.ReplaceSubstructs(
        single_conf,
        Chem.AddHs(Chem.MolFromSmarts("[NH3]")),
        Chem.MolFromSmarts("N#N"),
        replaceAll=True,
    )[0]

    # Get bare Mo to use as embed reference
    Mo_3d = remove_NH3(single_conf)
    Mo_3d = Chem.AddHs(Mo_3d)

    # Change charge of the N bound to Mo to ensure sanitation works
    match = Mo_N2.GetSubstructMatch(Chem.MolFromSmarts("[Mo]N#N"))
    Mo_N2.GetAtomWithIdx(match[1]).SetFormalCharge(1)

    # Embed catalyst
    if scoring_args.get("ga_scoring", False):
        Mo_N2_3d = embed_rdkit(
            mol=Mo_N2, core=Mo_3d, numConfs=1, pruneRmsThresh=scoring_args["RMS_thresh"]
        )
    else:
        Mo_N2_3d = embed_rdkit(
            mol=Mo_N2,
            core=core[0],
            numConfs=scoring_args["n_confs"],
            pruneRmsThresh=scoring_args["RMS_thresh"],
        )

    with cd(scoring_args["output_dir"]):
        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_N2"
        scoring_args["charge"] = smi_dict["Mo_N2"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_N2"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_N2_3d, scoring_options=scoring_args)

        # Perform calculation
        optimized_mol2, energies2 = optimizer.optimize_schrock()

    confs = optimized_mol2.GetConformers()
    if len(confs) == 0:
        return None, None, {"energy1": None, "energy2": None, "score": np.nan}

    energies2, new_mol2 = energy_filter(confs, energies2, optimized_mol2, scoring_args)

    energy_diff = (
        ((energies2.min() + NH3_ENERGY) - (energies.min() + N2_ENERGY))
    ) * hartree2kcalmol
    print(f"Score for top scoring conformer: {energy_diff}")

    en_dict = {"energy1": energies, "energy2": energies2, "score": energy_diff}

    return new_mol, new_mol2, en_dict


def rdkit_embed_scoring_NH3plustoNH3(ligand, scoring_args):
    """Score the NH3+ -> NH3 charge transfer.

    Args:
        ligand (Chem.rdchem.Mol): ligand to put on Mo core
        scoring_args (dict): dict with all relevant args for xtb and general scoring
    """

    idx = ligand.idx

    # Unpack gas energies
    NH3_ENERGY, N2_ENERGY, CP_RED_ENERGY = GAS_ENERGIES[scoring_args["method"]]

    # Get mol object of Mo core + connected ligand
    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    Mo_NH3 = connect_ligand(core_NH3[0], ligand_cut, NH3_flag=True)
    Mo_NH3 = Chem.AddHs(Mo_NH3)

    # Embed mol object
    Mo_NH3_3d = embed_rdkit(
        mol=Mo_NH3,
        core=core_NH3[0],
        numConfs=scoring_args["n_confs"],
        pruneRmsThresh=scoring_args["RMS_thresh"],
    )

    with cd(scoring_args["output_dir"]):
        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3+"
        scoring_args["charge"] = smi_dict["Mo_NH3+"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_NH3+"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        optimized_mol1, energies = optimizer.optimize_schrock()

    confs = optimized_mol1.GetConformers()
    if len(confs) == 0:
        return None, None, {"energy1": None, "energy2": None, "score": np.nan}

    # Remove fonformers that are too far from the minimum energy conf
    energies, new_mol = energy_filter(confs, energies, optimized_mol1, scoring_args)

    single_conf = copy.deepcopy(new_mol)
    if scoring_args.get("ga_scoring", False):
        # Copy into new mol object
        # Remove higher energy conformes from mol object
        minidx = np.argmin(energies)
        discard_conf = [
            x for x in range(len(single_conf.GetConformers())) if x != minidx
        ]
        for elem in discard_conf:
            single_conf.RemoveConformer(elem)

    with cd(scoring_args["output_dir"]):
        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3"
        scoring_args["charge"] = smi_dict["Mo_NH3"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_NH3"]["spin"]
        optimizer = XTB_optimize_schrock(mol=single_conf, scoring_options=scoring_args)

        # Perform calculation
        optimized_mol2, energies2 = optimizer.optimize_schrock()

    confs = optimized_mol2.GetConformers()
    if len(confs) == 0:
        return None, None, {"energy1": None, "energy2": None, "score": np.nan}

    energies2, new_mol2 = energy_filter(confs, energies2, optimized_mol2, scoring_args)

    energy_diff = (energies2.min() - energies.min() + CP_RED_ENERGY) * hartree2kcalmol

    print(f"Score for top scoring conformer: {energy_diff}")

    en_dict = {"energy1": energies, "energy2": energies2, "score": energy_diff}

    return new_mol, new_mol2, en_dict
