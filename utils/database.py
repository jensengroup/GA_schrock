"""Module that contains mol manipulations and various resuable functionality
classes."""
import concurrent.futures
import os
import pickle
import re
import shutil
import sys
from glob import glob
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import read

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def extract_energyxtb(logfile=None):
    """Extracts xtb energies from xtb logfile using regex matching.

    Args:
        logfile (str): Specifies logfile to pull energy from

    Returns:
        energy (List[float]): List of floats containing the energy in each step
    """

    re_energy = re.compile("energy: (-\\d+\\.\\d+)")
    energy = []
    with logfile.open() as f:
        for line in f:
            if "energy" in line:
                energy.append(float(re_energy.search(line).groups()[0]))
    return energy


def write_to_db(args):
    """Write xtblog files to database file."""

    i, logfile = args

    trajs = read(logfile.parent / "traj.xyz", index=":")
    energies = extract_energyxtb(logfile)

    # Convert from hartree to eV.
    energies = [x * 27.2114 for x in energies]
    # Get length of loop
    length = len(energies)

    # FInd the corresponding GA_file
    idx = logfile.parents[1].name
    ga_idx = idx[0:3].strip("0")
    mol_idx = idx[4:7].strip("0")

    if not ga_idx:
        ga_idx = 0
    if not mol_idx:
        mol_idx = 0

    tmp = f"GA{ga_idx:0>2}"
    GA_file = logfile.parents[2] / f"{tmp}.pkl"

    # Get pickle ind object
    with open(GA_file, "rb") as f:
        try:
            ga_class = pickle.load(f)
        except pickle.PicklingError as exc:
            print("Got pickling error: {0}".format(exc))

    # Find mol with correct idx
    mol = [
        ind
        for ind in ga_class.molecules
        if (
            (ind.idx[1] == int(mol_idx))
            and ((ind.energy != 9999) and not (np.isnan(ind.energy)))
        )
    ]

    if mol:
        ind = mol[0]
        with connect(source / "ase.db", use_lock_file=False) as db:
            for k, (energy, struct) in enumerate(zip(energies, trajs)):
                entry_name = f"{logfile.parent}_{i}_{k}"
                id = db.reserve(name=entry_name)
                if id is None:
                    continue
                struct.calc = SinglePointCalculator(struct, energy=energy)

                # Lowest energy flag
                if k == length:
                    flag = True
                else:
                    flag = False
                db.write(
                    struct,
                    id=id,
                    name=entry_name,
                    data={
                        "smiles": ind.smiles,
                        "smiles": ind.smiles_sa,
                        "lowest_energy": flag,
                        "cut_idx": ind.cut_idx,
                    },
                )
    else:
        print("optimization did not meet conditions")
    return


def db_write_driver(ga_folder=None, workers=6):
    """Paralellize writing to db, not very robust."""

    database_dir = source / "ase.db"

    folder = Path(ga_folder)

    # Get paths to all log files, while excluding the force-fiels opts.
    logs = sorted(folder.rglob("[!ff]*.log"))

    [shutil.copy(x, x.parent / f"traj{i}.xyz") for i, x in enumerate(logs)]

    if logs:
        print(f"Found logs in {ga_folder}")
    args = [(i, traj) for i, traj in enumerate(logs)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(write_to_db, args)

    return


if __name__ == "__main__":

    # Get dir from command line
    dirs = str(sys.argv[1])
    workers = int(sys.argv[2])

    print(f"CONNECTING TO {source/'ase.db'}")
    k = sorted(glob(os.path.join(dirs, "*/")))

    for dir in k:
        try:
            db_write_driver(dir, workers)
        except Exception as e:
            print(e)
            print(f"Something failed for {dir}")

    db = connect(source / "ase.db")
    print("lol")
