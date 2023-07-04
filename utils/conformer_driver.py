import argparse
import logging
import os
import random
import sys
from ast import literal_eval as make_tuple
from pathlib import Path

import pandas as pd
from rdkit import Chem

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))

from dft.orca_driver import conformersearch_dft_driver

from scoring import scoring_functions as sc
from utils.classes import Conformers, Individual
from utils.utils import get_git_revision_short_hash

ORCA_COMMANDS = {
    "sp": "!PBE D3BJ ZORA ZORA-def2-TZVP SARC/J SPLIT-RI-J MiniPrint PrintMOs KDIIS SOSCF",
    "sp_sarcJ": "!PBE D3BJ ZORA ZORA-def2-TZVP  SARC/J SPLIT-RI-J MiniPrint PrintMOs KDIIS SOSCF",
    "opt": "!PBE D3BJ ZORA ZORA-def2-TZVP SARC/J SPLIT-RI-J MiniPrint PrintMOs KDIIS SOSCF OPT",
    "freq": "!PBE D3BJ ZORA ZORA-def2-SVP SARC/J SPLIT-RI-J NormalPrint KDIIS SOSCF FREQ",
    "final_sp": "!B3LYP D3BJ ZORA ZORA-def2-TZVP SARC/J SPLIT-RI-J RIJCOSX MiniPrint KDIIS SOSCF",
}


def get_arguments(arg_list=None):
    """

    Args:
        arg_list: Automatically obtained from the commandline if provided.
        Otherwise default arguments are used

    Returns:
        parser.parse_args(arg_list)(Namespace): Dictionary like class that contain the arguments

    """
    parser = argparse.ArgumentParser(
        description="Run conformer screeening", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random initialization of population",
    )
    parser.add_argument(
        "--n_confs",
        type=int,
        default=3,
        help="How many conformers to generate",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=2,
        help="Number of cores to distribute over",
    )
    parser.add_argument(
        "--RMS_thresh",
        type=float,
        default=0.25,
        help="RMS pruning in embedding",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="RMS pruning in embedding",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="xeon40",
        help="Which partition to run on",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="debug_conformer",
        help="Directory to put various files",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default="2",
        help="gfn method to use",
    )
    parser.add_argument("--bond_opt", action="store_true")
    parser.add_argument("--full_relax", action="store_true")
    # XTB specific params
    parser.add_argument(
        "--opt",
        type=str,
        default="tight",
        help="Opt convergence criteria for XTB",
    )
    parser.add_argument(
        "--gbsa",
        type=str,
        default="benzene",
        help="Type of solvent",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./xcontrol.inp",
        help="Name of input file that is created",
    )

    parser.add_argument(
        "--file",
        type=Path,
        default="data/second_conformers.csv",
        help="File to get mol objects from",
    )
    parser.add_argument("--write_db", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--dft", action="store_true")
    parser.add_argument(
        "--calc_dir",
        type=Path,
        default="debug_conformer",
        help="Path to folder containing xyz files",
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        default=40,
        help="How many cores for each calc",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=8,
        help="How many GB requested for each calc",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="niflheim",
        help="Which cluster the calc is running on",
    )
    parser.add_argument(
        "--no_molecules",
        type=int,
        default=[0, 10],
        nargs="+",
        help="How many of the top molecules to do DFT on",
    )
    parser.add_argument(
        "--type_calc",
        dest="type_calc",
        choices=list(ORCA_COMMANDS.keys()),
        required=True,
        help="""Choose top line for input file""",
    )
    parser.add_argument(
        "--energy_cutoff",
        type=float,
        default=0.0159,
        help="Cutoff for conformer energies in hartree",
    )
    return parser.parse_args(arg_list)


def get_start_population_from_csv(file=None):

    # Get ligands found with specified scoring function
    df = pd.read_csv(file, index_col=[0])

    mols = [Chem.MolFromSmiles(x) for x in df["smiles"]]

    # Match
    matches = [mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2]")) for mol in mols]

    # Get list of mol objects
    inds = [
        Individual(mol, cut_idx=random.choice(match)[0], idx=make_tuple(idx))
        for mol, match, idx in zip(mols, matches, df.index)
    ]

    for elem, scoring_function in zip(inds, df.scoring):
        elem.scoring_function = scoring_function

    # Initialize population object.
    conformers = Conformers(inds)

    return conformers


def get_start_population_debug(file=None):

    mols = [Chem.MolFromSmiles(x) for x in ["CCN", "CCCN"]]
    scoring = ["rdkit_embed_scoring_NH3toN2", "rdkit_embed_scoring_NH3toN2"]

    # Match
    matches = [mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2]")) for mol in mols]

    # Get list of mol objects
    inds = [
        Individual(mol, cut_idx=random.choice(match)[0], idx=make_tuple(idx))
        for mol, match, idx in zip(mols, matches, ["(0,1)", "(0,2)"])
    ]

    for elem, scoring_function in zip(inds, scoring):
        elem.scoring_function = scoring_function

    # Initialize population object.
    conformers = Conformers(inds)

    return conformers


def main():

    FUNCTION_MAP = {
        "conformer_dft": conformersearch_dft_driver,
    }

    # Get args
    args = get_arguments()

    # Create output folder
    args.output_dir.mkdir(exist_ok=True)

    # Get arguments as dict and add scoring function to dict.
    args_dict = vars(args)

    if args.debug:
        # Get start population from csv file
        conformers = get_start_population_debug(file=args.file)
        args_dict["opt"] = "loose"
    else:
        conformers = get_start_population_from_csv(file=args.file)

    # Create the scoring function dirs
    (args.output_dir / "rdkit_embed_scoring").mkdir(exist_ok=True, parents=True)
    (args.output_dir / "rdkit_embed_scoring_NH3toN2").mkdir(exist_ok=True, parents=True)
    (args.output_dir / "rdkit_embed_scoring_NH3plustoNH3").mkdir(
        exist_ok=True, parents=True
    )

    # Setup logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args_dict["output_dir"], "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),  # For debugging. Can be removed on remote
        ],
    )

    # Log current git commit
    logging.info("Current git hash: %s", get_git_revision_short_hash())

    # Log the argparse set values
    logging.info("Input args: %r", args)

    # Submit population for scoring with many conformers
    results = sc.slurm_scoring_conformers(conformers, args_dict)
    conformers.handle_results(results)

    # Save the results:
    conformers.save(directory=args.output_dir, name=f"Conformers.pkl")

    print("Done with XTB conformational search, Submitting DFT calcs")
    if args.dft:
        conformersearch_dft_driver(args)

    sys.exit(0)


if __name__ == "__main__":
    main()
