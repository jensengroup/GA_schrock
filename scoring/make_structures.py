# -*- coding: utf-8 -*-
"""Module that performs the handling mol ligands."""

import random

from rdkit import Chem
from rdkit.Chem import AllChem


def mol_with_atom_index(mol):
    """Visualize mol object with atom indices."""
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp(
            "molAtomMapNumber", str(mol.GetAtomWithIdx(idx).GetIdx())
        )
    Chem.Draw.MolToImage(mol, size=(900, 900)).show()
    return mol


def remove_NH3(mol):
    """Remove NH3 group on mol."""

    # Substructure match the NH3
    NH3_match = Chem.MolFromSmarts("[NH3]")
    NH3_match = Chem.AddHs(NH3_match)
    removed_mol = Chem.DeleteSubstructs(mol, NH3_match)

    return removed_mol


def remove_N2(mol):
    """Remove N2 group on mol."""

    # Substructure match the N2
    NH2_match = Chem.MolFromSmarts("N#N")
    removed_mol = Chem.DeleteSubstructs(mol, NH2_match)

    return removed_mol


def remove_dummy(mol):
    """Remove dummy atom from mol."""

    dum_match = Chem.MolFromSmiles("*")
    removed_mol = Chem.DeleteSubstructs(mol, dum_match)

    return removed_mol


def getAttachmentVector(mol, atom_num=0):
    """Search for the position of the attachment point and extract the atom index of the attachment point and the connected atom (only single neighbour supported)
    Function from https://pschmidtke.github.io/blog/rdkit/3d-editor/2021/01/23/grafting-fragments.html
    mol: rdkit molecule with a dummy atom
    return: atom indices
    """
    rindex = -1
    rindexNeighbor = -1
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == atom_num:
            rindex = atom.GetIdx()
            neighbours = atom.GetNeighbors()
            if len(neighbours) == 1:
                rindexNeighbor = neighbours[0].GetIdx()
            else:
                print("two attachment points not supported yet")
                return None

    return rindex, rindexNeighbor


def replaceAtom(mol, indexAtom, indexNeighbor, atom_type="Br"):
    """Replace an atom with another type."""

    emol = Chem.EditableMol(mol)
    emol.ReplaceAtom(indexAtom, Chem.Atom(atom_type))
    emol.RemoveBond(indexAtom, indexNeighbor)
    emol.AddBond(indexAtom, indexNeighbor, order=Chem.rdchem.BondType.SINGLE)
    return emol.GetMol()


def addAtom(mol, indexAtom, atom_type="N", order="single", flag=None):
    """Add atom and connect to indexAtom with specified order."""

    order_dict = {
        "single": Chem.rdchem.BondType.SINGLE,
        "double": Chem.rdchem.BondType.DOUBLE,
        "triple": Chem.rdchem.BondType.TRIPLE,
    }

    emol = Chem.EditableMol(mol)
    idx = emol.AddAtom(Chem.Atom(atom_type))
    emol.AddBond(indexAtom, idx, order=order_dict[order])

    f_mol = emol.GetMol()

    if flag:
        f_mol.GetAtomWithIdx(indexAtom).SetFormalCharge(1)

    return f_mol, idx


def atom_remover(mol, pattern=None):
    """Generator function that removes a substructures and yields all the n
    structures where each structure has one of the substructures removed.

    Args:
        mol (Chem.rdchem.Mol): The mol to remove substruct on
        pattern (Chem.rdchem.Mol): mol object to remove from the input mol

    Yields:
        Chem.rdchem.Mol: The ouput mol with 1 removed substructure
    """
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        yield Chem.Mol(mol)
    for match in matches:
        res = Chem.RWMol(mol)
        res.BeginBatchEdit()
        for aid in match:
            res.RemoveAtom(aid)
        res.CommitBatchEdit()
        Chem.SanitizeMol(res)
        yield res.GetMol()


def single_atom_remover(mol, idx):
    """Function that removes an atom at specified idx.

    Args:
        mol (Chem.rdchem.Mol): The mol to remove substruct on
        pattern (Chem.rdchem.Mol): mol object to remove from the input mol

    Returns:
        Chem.rdchem.Mol: The ouput mol with the atom removed
    """
    res = Chem.RWMol(mol)
    res.BeginBatchEdit()
    res.RemoveAtom(idx)
    res.CommitBatchEdit()
    Chem.SanitizeMol(res)
    return res.GetMol()


def connect_ligand(core, ligand, NH3_flag=None, N2_flag=None):
    """Function that takes two mol objects at creates a core with ligand.

    Args:
        core (Chem.rdchem.Mol): The core to put ligand on. With dummy atoms at
            ligand positions.
        ligand (Chem.rdchem.Mol): The ligand to put on core with dummy atom where
            N from the core should be.
        NH3_flag (bool): Flag to mark if the core has a NH3 on it.
            Then the charge is set to ensure non-faulty sanitation.
        N2_flag (bool): Flag to mark if the core has a N2 on it.
            Then the charge is set to ensure non-faulty sanitation.
    Returns:
        mol (Chem.rdchem.Mol): mol object with the ligand put on the core
    """

    # mol object for dummy atom to replace on the core
    dummy = Chem.MolFromSmiles("*")

    # Get idx of the dummy(idx) and the connecting atom(neigh_idx)
    idx, neigh_idx = getAttachmentVector(ligand)

    # Remove dummy
    attach = remove_dummy(ligand)

    # Put the ligand won the core with specified bonding atom in the ligand.
    mol = AllChem.ReplaceSubstructs(
        core,
        dummy,
        attach,
        replaceAll=True,
        replacementConnectionPoint=neigh_idx,
    )[0]

    if NH3_flag:
        match = mol.GetSubstructMatch(Chem.MolFromSmarts("[Mo][NH3]"))
        mol.GetAtomWithIdx(match[1]).SetFormalCharge(1)
    if N2_flag:
        match = mol.GetSubstructMatch(Chem.MolFromSmarts("[Mo]N#N"))
        mol.GetAtomWithIdx(match[1]).SetFormalCharge(1)

    # Sanitation ensures that it is a reasonable molecule.
    Chem.SanitizeMol(mol)

    # Ensure no coordinates exist yet.
    mol.RemoveAllConformers()

    return mol


def create_prim_amine(input_ligand):
    """A function that takes a ligand and splits on a nitrogen bond, and then
    returns a ligand that has a primary amine and a cut_idx that specifies the
    location of the primary amine.

    Args:
        input_ligand (Chem.rdchem.Mol): Ligand to modify

    Returns:
        lig (Chem.rdchem.Mol): Modified ligand with preferably only one primary amine attachment
            point.
        prim_amine_index tuple(tuple(int)): idx of a primary amine
    """

    # Initialize dummy mol
    dummy = Chem.MolFromSmiles("*")

    # Match Secondary or Tertiary amines
    matches = input_ligand.GetSubstructMatches(
        Chem.MolFromSmarts("[NX3;H1,H0;!$(*n);!$(*N)]")
    )
    if not matches:
        raise Exception(
            f"{Chem.MolToSmiles(Chem.RemoveHs(input_ligand))} constains no amines to split on"
        )

    # Check if all the amines are in a ring.
    matches = input_ligand.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H1,H0;!R]"))
    if not matches:
        print(
            "There are no non-ring amines, checking for methyl groups instead to use as attachment points"
        )

        methyl_matches = input_ligand.GetSubstructMatches(
            Chem.MolFromSmarts("[CX4;H3]")
        )
        # Shuffle list of matches
        m_list = list(methyl_matches)
        random.shuffle(m_list)
        for elem in m_list:
            # Add a nitrogen atom to the methyl and return
            atom = input_ligand.GetAtomWithIdx(elem[0])
            mol, idx = addAtom(input_ligand, elem[0])
            if mol:
                mol.UpdatePropertyCache()
                return mol, [[idx]]

    # Shuffle list of amine matches
    l = list(matches)
    random.shuffle(l)

    # Loop through matching amines and find one that works
    indices = []
    for match in l:
        # Get the atom object for the mathing atom
        atom = input_ligand.GetAtomWithIdx(match[0])

        # Create list of tuples that contain the amine idx and idx of each of the three
        # neighbors that are not another amine.
        banned_atoms = [7, 8]
        for x in atom.GetNeighbors():
            if x.GetAtomicNum() == 12:
                indices = [(match[0], x.GetIdx())]
            elif (x.GetAtomicNum() not in banned_atoms) and not (
                input_ligand.GetBondBetweenAtoms(match[0], x.GetIdx()).IsInRing()
            ):
                indices = [(match[0], x.GetIdx())]

        # Break loop if a valid match is found
        if indices:
            break

    # If not indices were found None is returned
    try:
        atoms = random.choice(indices)
    except IndexError as e:
        print("Oh no, found no valid cut points")
        return None, None
    # Get the bond idx of the chosen match
    bond = [input_ligand.GetBondBetweenAtoms(*atoms).GetIdx()]

    # Get the fragments from breaking the amine bonds.
    # OBS! If the fragments connected to the tertiary amine, are connected
    # then the resulting ligand will have multiple dummy locations which will break
    # the workflow
    frag = Chem.FragmentOnBonds(
        input_ligand, bond, addDummies=True, dummyLabels=[(1, 1)]
    )
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

    # Select the fragment that was cut from amine.
    # If there is only one fragment, it can break so I added the temporary
    # ff statement
    if len(frags) == 1:
        ligand = [frags[0]]
    else:
        ligand = [
            struct
            for struct in frags
            if not struct.HasSubstructMatch(Chem.MolFromSmarts("[1*]N"))
        ]

    # As this function is also run outside paralellization, an error here will break
    # the whole driver. This statement ensures that something is returned at least
    # If the list is empty.
    if not ligand:
        return None, None

    # Put primary amine on the dummy location for the ligand just created.
    N_mol = Chem.MolFromSmiles("N")
    lig = AllChem.ReplaceSubstructs(
        ligand[0], dummy, N_mol, replacementConnectionPoint=0, replaceAll=True
    )[0]

    # Get idx where to cut.
    prim_amine_index = lig.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2]"))
    if len(prim_amine_index) > 1:
        print(
            f"There are several primary amines to cut at with idxs: {prim_amine_index}"
            f"removing some"
        )
        # Substructure match primary amine
        prim_match = Chem.MolFromSmarts("[NX3;H2]")

        # Remove the primary amines and chose one of the structures
        ms = [x for x in atom_remover(lig, pattern=prim_match)]
        lig = random.choice(ms)

        # Get the idx of the remaining amine
        prim_amine_index = lig.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2]"))

    # Need this to prevent errors later. See: https://github.com/rdkit/rdkit/issues/1596
    output_ligand = Chem.MolFromSmiles(Chem.MolToSmiles(lig))
    output_ligand.UpdatePropertyCache()
    prim_amine_index = output_ligand.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2]"))

    # Last error check step
    if not prim_amine_index:
        print(
            f"Something was wrong for this molecule with smiles {Chem.MolToSmiles(input_ligand)}"
        )
        prim_amine_index = [[1]]

    return output_ligand, prim_amine_index


def create_dummy_ligand(ligand, cut_idx=None):
    """Cut atom from ligand and put dummy idx.

    Args:
        ligand (mol): ligand to remove atom from
        cut_idx (int): index of atom to remove

    Returns:
        ligands (Chem.rdchem.Mol) : ligand with dummy atom
    """

    # Initialize dummy mol
    dummy = Chem.MolFromSmiles("*")

    # Get the neighbouring bonds to the amine given by cut_idx
    atom = ligand.GetAtomWithIdx(cut_idx)

    # Create list of tuples that contain the amine idx and idx of neighbor.
    indices = [
        (cut_idx, x.GetIdx()) for x in atom.GetNeighbors() if x.GetAtomicNum() != 1
    ][0]

    # Get the bonds to the neighbors.
    bond = []
    bond.append(ligand.GetBondBetweenAtoms(indices[0], indices[1]).GetIdx())

    # Get the two fragments, the ligand and the NH2
    frag = Chem.FragmentOnBonds(ligand, bond, addDummies=True, dummyLabels=[(1, 1)])
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

    # Pattern for N connected to dummy
    smart = "[NX3;H2][1*]"
    patt = Chem.MolFromSmarts(smart)

    # Get the ligand that is not NH2
    ligands = [struct for struct in frags if not struct.GetSubstructMatches(patt)]

    return ligands[0]


def embed_rdkit(
    mol,
    core,
    numConfs=1,
    coreConfId=-1,
    randomseed=2342,
    numThreads=1,
    force_constant=1e6,
    pruneRmsThresh=-1,
):
    """Embedding driver function.

    Args:
        mol (Mol): Core+ligand mol object.
        core (Mol): Core with dummy atoms on ligand positions
        numConfs (int): How many conformers to get from embedding
        coreConfId (int): If core has multiple conformers this indicates which one to choose
        randomseed (int): Seed for embedding.
        numThreads (int): How many threads to use for embedding.
        force_constant (float): For alignment
        pruneRmsThresh (int): Embedding parameter

    Returns:
        mol (Chem.rdchem.Mol): Embedded mol object
    """
    # Match the core+ligand to the Mo core.
    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")

    # Get the coordinates for the core, which constrains the embedding
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    # Embed the mol object constrained to the core.
    cids = AllChem.EmbedMultipleConfs(
        mol=mol,
        numConfs=numConfs,
        coordMap=coordMap,
        maxAttempts=10,
        randomSeed=2,
        numThreads=numThreads,
        pruneRmsThresh=pruneRmsThresh,
        useRandomCoords=True,
    )

    cids = list(cids)

    # If embedding failed, retry with other parameters.
    # ignoreSmoothingFailures removed the majority of embedding errors for me
    if not cids:
        cids = AllChem.EmbedMultipleConfs(
            mol=mol,
            numConfs=numConfs,
            coordMap=coordMap,
            maxAttempts=10,
            randomSeed=3,
            numThreads=numThreads,
            pruneRmsThresh=pruneRmsThresh,
            useRandomCoords=True,
            ignoreSmoothingFailures=True,
        )
        Chem.SanitizeMol(mol)
        if not cids:
            print(coordMap, Chem.MolToSmiles(mol))
            raise ValueError("Could not embed molecule")

    # Rotate embedded conformations onto the core
    algMap = [(j, i) for i, j in enumerate(match)]
    for cid in cids:
        rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)

    return mol


if __name__ == "__main__":
    print("Hello, nothing here :)")
