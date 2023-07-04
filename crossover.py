# %%
"""Written by Jan H.

Jensen 2018
"""
import random

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

rdBase.DisableLog("rdApp.error")


def cut(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[*]-;!@[*]")):
        return None
    bis = random.choice(
        mol.GetSubstructMatches(Chem.MolFromSmarts("[*]-;!@[*]"))
    )  # single bond not in ring
    # print bis,bis[0],bis[1]
    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]

    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

    try:
        fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
        return fragments
    except:
        return None


def cut_ring(mol):
    for i in range(10):
        if random.random() < 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")):
                return None
            bis = random.choice(
                mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]@[R]@[R]"))
            )
            bis = (
                (bis[0], bis[1]),
                (bis[2], bis[3]),
            )
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")):
                return None
            bis = random.choice(
                mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R;!D2]@[R]"))
            )
            bis = (
                (bis[0], bis[1]),
                (bis[1], bis[2]),
            )

        # print bis
        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]

        fragments_mol = Chem.FragmentOnBonds(
            mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)]
        )

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
        except:
            return None

        if len(fragments) == 2:
            return fragments

    return None


def ring_OK(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(
        Chem.MolFromSmarts("[r3,r4]=[r3,r4]")
    )

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


def tert_amine_OK(mol):
    """Checks if there is at least on tertiary amine."""
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H0;D3;!$(NC=O);!$(*n);!$(*N)]")):
        return True
    else:
        return False


def primary_secondary_amine_OK(mol):
    """Checks if there is any secondary or primary amines."""
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O);!$(*n);!$(*N)]")):
        return True
    else:
        return False


def is_parameterized(mol, forcefield="UFF"):
    if forcefield is "UFF":
        return AllChem.UFFHasAllMoleculeParams(mol)
    if forcefield is "MMFF":
        return AllChem.MMFFHasAllMoleculeParams(mol)


def mol_issane(mol: Chem.Mol, filter) -> bool:
    """Checks that a RDKit molecule matches some filter If a match is found
    between the molecule and the filter the molecule is NOT suitable for
    further use.

    :param mol: SMILES string of molecule
    :param filter: a frame with a filter
    """
    # always return True (molecule OK) if a filter is not supplied
    if filter is None:
        return True

    for pattern in filter:
        if mol.HasSubstructMatch(pattern):
            # print(smarts, row['rule_set_name']) #debug
            print("matches:", Chem.MolToSmarts(pattern))
            return False

    return True


def mol_OK_old(mol, molecule_filter):
    if not size_stdev or not average_size:
        print("size parameters are not defined")
    try:
        Chem.SanitizeMol(mol)
        test_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if test_mol == None:
            return None
        if not mol_is_sane(mol, molecule_filter):
            return False
        target_size = (
            size_stdev * np.random.randn() + average_size
        )  # parameters set in GA_mol
        target_nrb = 2 * np.random.randn() + 5
        if target_nrb < 5:
            target_nrb = 5
        print(CalcNumRotatableBonds(mol))
        if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
            return True
        else:
            return False
    except:
        return False


def mol_OK(mol, filter):
    """Returns of molecule on input is OK according to various criteria
    Criteria currently tested are:

      * check if RDKit can understand the smiles string
      * check if the size is OK
      * check if the molecule is sane
    :param mol string: SMILES string
    :param filter string: the name of the filter to use
    """
    try:
        # check RDKit understands a molecule
        Chem.SanitizeMol(mol)
        test_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if test_mol is None:
            return False

        # check molecule is sane
        if not mol_issane(mol, filter):
            return False

        # check molecule size
        target_size = (
            size_stdev * np.random.randn() + average_size
        )  # parameters set in GA_mol
        if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
            return True
        else:
            return False
    except:
        return False


def crossover_ring(parent_A, parent_B, filter):
    ring_smarts = Chem.MolFromSmarts("[R]")
    if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(
        ring_smarts
    ):
        return None

    rxn_smarts1 = [
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]",
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]",
    ]
    rxn_smarts2 = [
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]",
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]",
    ]
    for i in range(10):
        fragments_A = cut_ring(parent_A)
        fragments_B = cut_ring(parent_B)
        # print [Chem.MolToSmiles(x) for x in list(fragments_A)+list(fragments_B)]
        if fragments_A == None or fragments_B == None:
            return None

        new_mol_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            new_mol_trial = []
            for fa in fragments_A:
                for fb in fragments_B:
                    new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])

        new_mols = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_OK(m, filter):
                    new_mols += list(rxn2.RunReactants((m,)))

        new_mols2 = []
        for m in new_mols:
            m = m[0]
            if (
                mol_OK(m, filter)
                and ring_OK(m)
                and (primary_secondary_amine_OK(m) or tert_amine_OK(m))
            ):
                new_mols2.append(m)

        if len(new_mols2) > 0:
            return random.choice(new_mols2)

    return None


def crossover_non_ring(parent_A, parent_B, filter):
    for i in range(10):
        fragments_A = cut(parent_A)
        fragments_B = cut(parent_B)
        if fragments_A == None or fragments_B == None:
            return None
        rxn = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
        new_mol_trial = []
        for fa in fragments_A:
            for fb in fragments_B:
                new_mol_trial.append(rxn.RunReactants((fa, fb))[0])
        new_mols = []
        for mol in new_mol_trial:
            mol = mol[0]
            if mol_OK(mol, filter) and (
                primary_secondary_amine_OK(mol) or tert_amine_OK(mol)
            ):
                new_mols.append(mol)

        if len(new_mols) > 0:
            return random.choice(new_mols)

    return None


def crossover(parent_A, parent_B, filter):
    parent_smiles = [Chem.MolToSmiles(parent_A), Chem.MolToSmiles(parent_B)]
    try:
        Chem.Kekulize(parent_A, clearAromaticFlags=True)
        Chem.Kekulize(parent_B, clearAromaticFlags=True)
    except:
        pass
    for i in range(10):
        if random.random() <= 0.5:
            # print 'non-ring crossover'
            new_mol = crossover_non_ring(parent_A, parent_B, filter)
            if new_mol != None:
                new_smiles = Chem.MolToSmiles(new_mol)
            if new_mol != None and new_smiles not in parent_smiles:
                return new_mol
        else:
            # print 'ring crossover'
            new_mol = crossover_ring(parent_A, parent_B, filter)
            if new_mol != None:
                new_smiles = Chem.MolToSmiles(new_mol)
            if new_mol != None and new_smiles not in parent_smiles:
                return new_mol

    return new_mol


if __name__ == "__main__":

    smiles1 = "Cc1ccc(S(=O)(=O)N2C(N)=C(C#N)C(c3ccc(Cl)cc3)C2C(=O)c2ccccc2)cc1"
    smiles2 = "C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1"

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    child = crossover(mol1, mol2, None)
