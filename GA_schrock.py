"""Driver script for running the genetic algorithm on the Schrock catalyst.

Written by Magnus Strandgaard 2023

Example:
    How to run:

        $ python GA_schrock.py --args
"""

import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path

import crossover as co
import filters
import GB_GA as ga
from scoring import scoring_functions as sc
from scoring.scoring import (
    rdkit_embed_scoring,
    rdkit_embed_scoring_NH3plustoNH3,
    rdkit_embed_scoring_NH3toN2,
)
from utils.classes import Generation
from utils.utils import get_git_revision_short_hash

molecule_filter = filters.get_molecule_filters(None, "./filters/alert_collection.csv")


def get_arguments(arg_list=None):
    """

    Args:
        arg_list: Automatically obtained from the commandline if provided.
        Otherwise default arguments are used

    Returns:
        parser.parse_args(arg_list)(Namespace): Dictionary like class that contain the driver arguments.

    """
    parser = argparse.ArgumentParser(
        description="Run GA algorithm", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=10,
        help="Sets the size of population pool.",
    )
    parser.add_argument(
        "--mating_pool_size",
        type=int,
        default=3,
        help="Size of mating pool",
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
        default=2,
        help="How many conformers to generate",
    )
    parser.add_argument(
        "--n_tries",
        type=int,
        default=1,
        help="How many overall runs of the GA",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=2,
        help="Number of cores to distribute xTB over",
    )
    parser.add_argument(
        "--RMS_thresh",
        type=float,
        default=0.25,
        help="RMS pruning in embedding",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=1,
        help="How many times is the population optimized",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.5,
        help="Probability of mutation of new children",
    )
    parser.add_argument(
        "--sa_screening",
        dest="sa_screening",
        default=False,
        action="store_true",
        help="Activates score scaling with synthetic accessibility",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="data/ZINC_250k.smi",
        help="The data used to create the starting database. Needs to be a \
        file with smiles",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generation_debug",
        help="Directory to put all results/output files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=12,
        help="Slurm timeout for scoring jobs.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="kemi1",
        help="Which cluster partition to run scoring on",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to do faster calculations on small population for debugging",
    )
    parser.add_argument(
        "--ga_scoring",
        action="store_true",
        help="Flag used to distinguish conformer search scoring and GA scoring",
    )
    parser.add_argument(
        "--supress_amines",
        action="store_true",
        help="Flag to remove extra primary amines during GA runs",
    )
    parser.add_argument(
        "--energy_cutoff",
        type=float,
        default=0.0159,
        help="Cutoff for conformer energies relative to the lowest conformer during GA runs",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Flag to delete calculation folders after runs. Pickles with GA generation data are still saved",
    )
    parser.add_argument(
        "--scoring_func",
        dest="func",
        choices=[
            "rdkit_embed_scoring",
            "rdkit_embed_scoring_NH3toN2",
            "rdkit_embed_scoring_NH3plustoNH3",
        ],
        required=True,
        help="""Choose one of the specified scoring functions to be run.""",
    )
    # XTB specific params
    parser.add_argument(
        "--method",
        type=str,
        default="2",
        help="gfn method to use",
    )
    parser.add_argument("--bond_opt", action="store_true")
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
        help="Name of xTB input file that is created",
    )
    parser.add_argument(
        "--average_size",
        type=int,
        default=12,
        help="Average number of heavy atoms resulting from crossover",
    )
    parser.add_argument(
        "--size_stdev",
        type=int,
        default="3",
        help="STD of crossover molecule size distribution",
    )
    return parser.parse_args(arg_list)


def GA(args):
    """

    Args:
        args(dict): Dictionary containing all the commandline input args.

    Returns:
        gen: Generation class that contains the results of the final generation
    """

    # Create initial population and get initial score. Option for debugging.
    if args["debug"]:
        population = ga.make_initial_population_debug(
            population_size=args["population_size"]
        )
    else:
        population = ga.make_initial_population(
            args["population_size"], args["file_name"]
        )

    # Score initial population
    results = sc.slurm_scoring(args["scoring_function"], population, args)

    # Set results on population class and do some rdkit hack to prevent weird molecules
    population.handle_results(results)
    population.update_property_cache()

    # Save current population for debugging
    population.save(directory=args["output_dir"], name="GA_debug_firstit.pkl")

    # Functionality to scale score with synthetic accessibility
    if args["sa_screening"]:
        population.get_sa()

    # Reweight score by rotatable bonds
    population.reweigh_rotatable_bonds()

    # Normalize the score of population individuals to value between 0 and 1
    population.sortby("score")
    population.calculate_normalized_fitness()

    # Save the generation as pickle file and print current output
    population.save(directory=args["output_dir"], name="GA00.pkl")
    logging.info(
        f"\n --- Current top generation for gen_no 0 ---\n {population.get_text()}"
    )
    with open(args["output_dir"] + "/GA0.out", "w") as f:
        f.write(population.get_text(population="molecules") + "\n")
        f.write(population.print_fails())

    logging.info("Finished initial generation")

    # Start evolving
    for generation in range(args["generations"]):
        # Counter for tracking generation number
        generation_num = generation + 1
        logging.info("Starting generation %d", generation_num)

        # Ensure no weird RDKit erorrs
        population.update_property_cache()

        # Get mating pool
        mating_pool = ga.make_mating_pool(population, args["mating_pool_size"])

        # If debugging simply reuse previous pop
        if args["debug"]:
            new_population = Generation(
                generation_num=generation_num,
                molecules=population.molecules,
            )
        else:
            new_population = ga.reproduce(
                mating_pool,
                args["population_size"],
                args["mutation_rate"],
                molecule_filter=molecule_filter,
            )

        # Save current population for debugging
        new_population.save(
            directory=args["output_dir"], name=f"GA{generation_num:02d}_debug.pkl"
        )

        logging.info("Creating attachment points for new population")

        # Process population to ensure primary amine attachment points
        new_population.modify_population(supress_amines=True)

        # Assign generation and population idx to the population
        new_population.generation_num = generation_num
        new_population.assign_idx()

        # Calculate new scores
        logging.info("Getting scores for new population")
        results = sc.slurm_scoring(args["scoring_function"], new_population, args)

        new_population.handle_results(results)
        new_population.save(
            directory=args["output_dir"], name=f"GA{generation_num:02d}_debug2.pkl"
        )

        # Functionality to compute synthetic accessibility
        if args["sa_screening"]:
            new_population.get_sa()

        # Reweight by rotatable bonds
        new_population.reweigh_rotatable_bonds()

        new_population.sortby("score")

        # Create tmp population from current best molecules
        potential_survivors = copy.deepcopy(population.molecules)

        # The calculated population is merged with current top population
        population = ga.sanitize(
            potential_survivors + new_population.molecules, args["population_size"]
        )

        population.generation_num = generation_num

        # Normalize new scores to prep for next gen
        population.calculate_normalized_fitness()

        # Collect result molecules in class.
        current_gen = Generation(
            generation_num=generation_num,
            molecules=population.molecules,
            new_molecules=new_population.molecules,
        )
        # Save data from current generation
        logging.info("Saving current generation")
        current_gen.save(
            directory=args["output_dir"], name=f"GA{generation_num:02d}.pkl"
        )

        # Print to individual generation files to keep track on the fly
        logging.info(
            f"\n --- Current top generation for gen_no {generation_num} ---\n {population.get_text()}"
        )
        with open(args["output_dir"] + f"/GA{generation_num}.out", "w") as f:
            f.write(current_gen.get_text(population="molecules") + "\n")
            f.write(current_gen.get_text(population="new_molecules") + "\n")
            f.write(current_gen.print_fails())

    return current_gen


def main():
    """Main function that starts the GA."""
    args = get_arguments()
    funcs = {
        "rdkit_embed_scoring": rdkit_embed_scoring,
        "rdkit_embed_scoring_NH3toN2": rdkit_embed_scoring_NH3toN2,
        "rdkit_embed_scoring_NH3plustoNH3": rdkit_embed_scoring_NH3plustoNH3,
    }

    # Get arguments as dict and add scoring function to dict.
    args_dict = vars(args)
    args_dict["scoring_function"] = funcs[args.func]

    # Create list of dicts for the distributed GAs
    GA_args = args_dict

    # Variables for crossover module
    co.average_size = args.average_size
    co.size_stdev = args.size_stdev

    # Run the GA
    for i in range(args.n_tries):
        # Start the time
        t0 = time.time()
        # Create output_dir
        GA_args["output_dir"] = args_dict["output_dir"] + f"_{i}"
        Path(GA_args["output_dir"]).mkdir(parents=True, exist_ok=True)

        # Setup logger
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(GA_args["output_dir"], "printlog.txt"), mode="w"
                ),
                logging.StreamHandler(),  # For debugging. Can be removed on remote
            ],
        )

        # Log current git commit hash
        logging.info("Current git hash: %s", get_git_revision_short_hash())

        # Log the argparse set values
        logging.info("Input args: %r", args)
        generations = GA(GA_args)

        # Final output handling and logging
        t1 = time.time()
        logging.info(f"# Total duration: {(t1 - t0) / 60.0:.2f} minutes")

    # Ensure the program exists when running on the frontend.
    sys.exit(0)


if __name__ == "__main__":
    main()
