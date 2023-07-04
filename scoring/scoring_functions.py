# -*- coding: utf-8 -*-
"""Module that contains submitit functionality to submit scoring function."""

import shutil
from pathlib import Path

import numpy as np
import submitit

from scoring.scoring import scoring_submitter


def slurm_scoring(sc_function, population, scoring_args):
    """Evaluates a scoring function for population on SLURM cluster.

    Args:
        sc_function (function): Scoring function to use for each molecule
        population List(Individual): List of molecules objects to score
        scoring_args (dict): Relevant scoring args for submitit or XTB
    Returns:
        results List(tuple): List of tuples containing result for each molecule
    """

    # Initialize AutoExecutor
    executor = submitit.AutoExecutor(
        folder=Path(scoring_args["output_dir"]) / "scoring_tmp",
        slurm_max_num_timeout=0,
    )
    executor.update_parameters(
        name=f"sc_g{population.molecules[0].idx[0]}",
        cpus_per_task=scoring_args["cpus_per_task"],
        slurm_mem_per_cpu="500MB",
        timeout_min=scoring_args["timeout"],
        slurm_partition=scoring_args["partition"],
        slurm_array_parallelism=100,
    )

    jobs = executor.map_array(
        sc_function, population.molecules, [scoring_args for p in population.molecules]
    )

    # Get the jobs results. Assign None variables if an error is returned for the given molecule
    results = [
        catch(
            job.result,
            handle=lambda e: (
                None,
                None,
                {"energy1": None, "energy2": None, "score": np.nan},
            ),
        )
        for job in jobs
    ]

    # Remove slurm log dir
    if scoring_args["cleanup"]:
        shutil.rmtree(Path(scoring_args["output_dir"]) / "scoring_tmp")

    return results


def slurm_scoring_conformers(conformers, scoring_args):
    """Evaluates a scoring function for population on SLURM cluster.

    Args:
        conformers List(Individual): List of molecules objects to do conformers search for
        scoring_args (dict): Relevant scoring args for submitit or XTB
    Returns:
        results List(tuple): List of tuples containing result for each molecule
    """
    executor = submitit.AutoExecutor(
        folder=Path(scoring_args["output_dir"]) / "scoring_tmp",
        slurm_max_num_timeout=0,
    )
    mem_per_cpu = (scoring_args["memory"] * 1000) // scoring_args["cpus_per_task"]
    executor.update_parameters(
        name=f"conformer_search",
        cpus_per_task=scoring_args["cpus_per_task"],
        slurm_mem_per_cpu=f"{mem_per_cpu}MB",
        timeout_min=scoring_args["timeout"],
        slurm_partition=scoring_args["partition"],
        slurm_array_parallelism=20,
    )

    jobs = executor.map_array(
        scoring_submitter,
        conformers.molecules,
        [scoring_args for c in conformers.molecules],
    )

    # Get the jobs results. Assign None variables if an error is returned for the given molecule
    results = [
        catch(
            job.result,
            handle=lambda e: (
                None,
                None,
                {"energy1": None, "energy2": None, "score": np.nan},
            ),
        )
        for job in jobs
    ]

    if scoring_args["cleanup"]:
        shutil.rmtree(Path(scoring_args["output_dir"]) / "scoring_tmp")

    return results


def catch(func, *args, handle=lambda e: e, **kwargs):
    """Helper function that takes the submitit result and returns an exception
    if no results can be retrieved."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        return handle(e)


### MolSimplify
# Submitit scoring functions related to molSimplify driver scripts


def slurm_molS(sc_function, scoring_args):
    """To submit create_cycle_MS to the commandline and create all Mo
    intermediates with a given ligand.

    Args:
        sc_function (func): molS driver function
        scoring_args (dict): Relevant scoring args

    Returns:
        results List(tuples): Resulst of molS output, not used.
    """
    executor = submitit.AutoExecutor(
        folder=Path(scoring_args["run_dir"]) / "scoring_tmp",
        slurm_max_num_timeout=0,
    )
    executor.update_parameters(
        name=f"cycle",
        cpus_per_task=scoring_args["ncores"],
        slurm_mem_per_cpu="2GB",
        timeout_min=10,
        slurm_partition=scoring_args["partition"],
        slurm_array_parallelism=2,
    )

    job = executor.submit(sc_function, **scoring_args)

    results = catch(job.result, handle=lambda e: None)

    return results
