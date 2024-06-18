# GB-GA for the Schrock catalyst

Repository for the paper [Genetic algorithm-based re-optimization of the Schrock catalyst for dinitrogen fixation](https://peerj.com/articles/pchem-30/).
The GA implementation is based on the [GB_GA](https://github.com/jensengroup/GB_GA).

### Note

This codebase is slightly outdated for the work we do on catalyst design with GAs. For the interested, this is a more robust codebase we are currently using for catalyst design: [catalystGA](https://github.com/juius/catalystGA). And for other updated work on the Schrock catalyst see: [genetic_algorithm_for_nitrogen_fixation](https://github.com/jensengroup/genetic_algorithm_for_nitrogen_fixation/tree/main)

1. [How to run](#how-to-run)
1. [Driver function arguments](driver-function-arguments)

## How to run

For simple use of the GA install with conda install the env file.

```
conda env create --file environment.yml
```

To run the GA activate the relevant environment and run the following for a quick run on a local installation:

```
python GA_schrock.py --supress_amines --debug --average_size 10 --size_stdev 2 --scoring_func rdkit_embed_scoring_NH3toN2 --cpus_per_task 2 --population_size 3 --mating_pool_size 4
```

### Driver function arguments

A list of possible commandline arguments.

| Arg                  | Description                                                                                                          |
| -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `-h` or `--help`     | Prints help message.                                                                                                 |
| `--population_size`  | Sets the size of the population pool.                                                                                |
| `--mating_pool_size` | Sets the size of the mating pool.                                                                                    |
| `--n_confs`          | Sets how many conformers to generate for each molecule.                                                              |
| `--n_tries`          | Sets how many times the GA is restarted. Can be used to run multiple GA runs in a single submission.                 |
| `--cpus_per_task`    | How many cores to use for each scoring job.                                                                          |
| `--RMS_thresh`       | RMS cutoff value for RDKit conformer embedding.                                                                      |
| `--generations`      | How many evolution cycles of the population is performed.                                                            |
| `--mutation_rate`    | Decides the probability of performing a mutation operation instead of crossover.                                     |
| `--sa_screening`     | Decides if synthetic accessibility score is enabled. Highly recommended to turn this on.                             |
| `--file_name`        | Path to the database extract to create starting population.                                                          |
| `--output_dir`       | Sets output directory for all files generated during generations.                                                    |
| `--timeout`          | How many minutes each slurm job is allowed to run.                                                                   |
| `--debug`            | If set the starting population is a set of 4 small molecules that can run fast locally. Used for debugging.          |
| `--ga_scoring`       | If set, removes all higher energy conformers in GA.                                                                  |
| `--supress_amines`   | Supress amine heavy molecules by converting any primary amines to hydrogen in generations.                           |
| `--method`           | Which gfn method to use.                                                                                             |
| `--energy_cutoff`    | Sets energy cutoff on the conformer filtering.                                                                       |
| `--bond_opt`         | Decides if a final Mo-N bond optimization is performed during scoring.                                               |
| `--cleanup`          | If enabled, all scoring files are removed after scoring. Only the optimized structures and their energies are saved. |
| `--scoring_func`     | Which scoring function to use.                                                                                       |
| `--opt`              | Set optimization convergence criteria for xTB.                                                                       |
| `--gbsa`             | Which type of solvent to use for xTB.                                                                                |
| `--input`            | Name of input control file created for xTB                                                                           |
| `--average_size`     | Average number of atoms in molecules resulting from crossover.                                                       |
| `--size-stdev`       | STD of crossover size distribution for molecules.                                                                    |

# Authors

**Magnus Strandgaard**<sup>1</sup>
**Julius Seumer**<sup>1</sup>
**Jan H. Jensen**<sup>1</sup>

<sup>1</sup> Department of Chemistry, University of Copenhagen, 2100 Copenhagen Ø, Denmark.

For any questions regarding the code, please do not hesitate to contact me at : _mastr@chem.ku.dk_.
