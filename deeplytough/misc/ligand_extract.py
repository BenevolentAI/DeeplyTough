import os

import Bio.PDB as PDB
import numpy as np

from misc.cc_ligands import ignore_list
from misc.utils import NonUniqueStructureBuilder


def residue_dist_to_ligand(protein_residue, ligand_residue):
    """Returns distance from the protein to the closest non-hydrogen ligand atom. This seems to be consistent with
    what pdbbind is using"""
    distances = []
    for lig_atom in ligand_residue.child_list:
        if not lig_atom.element or lig_atom.element[0] != 'H':
            for prot_atom in protein_residue.child_list:
                diff_vector = prot_atom.coord - lig_atom.coord
                distances.append(np.sqrt(np.sum(diff_vector * diff_vector)))
    return min(distances) if len(distances) else 1e10


class NearLigandSelect(PDB.Select):
    def __init__(self, distance_threshold, ligand_residue, keep_lig_in_site, keep_water, keep_other_hets=False):
        self.threshold = distance_threshold
        self.ligand_res = ligand_residue
        self.keep_water = keep_water
        self.keep_lig = keep_lig_in_site
        self.keep_other_hets = keep_other_hets

    def accept_residue(self, residue):
        if residue == self.ligand_res:
            return self.keep_lig
        elif not self.keep_other_hets and residue.get_id()[0].strip() != '':
            return False
        else:
            if not self.keep_water:
                if residue.resname == 'HOH':
                    return False
            dist = residue_dist_to_ligand(residue, self.ligand_res)
            return dist < self.threshold


class LigandOnlySelect(PDB.Select):
    def __init__(self, ligand_residue):
        self.ligand_residue = ligand_residue

    def accept_residue(self, residue):
        if residue == self.ligand_residue:
            return True  # change this to False if you don't want the ligand
        else:
            return False


class ChainOnlySelect(PDB.Select):
    def accept_residue(self, residue):
        return residue.get_id()[0].strip() == ''


def filter_unwanted_het_ids(het_list):
    het_list = filter(lambda x: x not in ignore_list, het_list)
    return list(het_list)


def get_het_residues_from_pdb(model, remove_duplicates=False, min_lig_atoms=-1, allowed_names=None):
    resi_list = []
    res_names = []
    for res in model.get_residues():
        if res._id[0].startswith('H_'):
            if remove_duplicates and res.resname in res_names:
                continue
            if min_lig_atoms > 0 and len(list(res.get_atoms())) < min_lig_atoms:
                continue
            if allowed_names and res.resname not in allowed_names:
                continue
            resi_list.append(res)
            res_names.append(res.resname)
    wanted_res_names = filter_unwanted_het_ids(set(res_names))
    wanted_resi_list = []
    for res in resi_list:
        if res.resname in wanted_res_names:
            wanted_resi_list.append(res)
    return wanted_resi_list


class PocketFromLigandDetector:
    """
    Extracts pockets around a ligand (which is either part of the input PDB file, or already separated in
    a different file).
    """
    def __init__(self, distance_threshold=8.0, ligand_fname_pattern=None, include_het_resname=True,
                 save_clean_structure=False, keep_other_hets=True, min_lig_atoms=-1, allowed_lig_names=None):
        """

        :param distance_threshold: Max distance between residue and ligand
        :param ligand_fname_pattern: A tuple (old, new) used to obtain ligand's file name by replacing `old` with `new`
        """

        self.distance_threshold = distance_threshold
        self.ligand_fname_pattern = ('', '') if ligand_fname_pattern is None else ligand_fname_pattern
        self.keep_water = False
        self.include_het_resname = include_het_resname
        self.save_clean_structure = save_clean_structure
        self.keep_other_hets = keep_other_hets
        self.min_lig_atoms = min_lig_atoms
        self.allowed_lig_names = allowed_lig_names

    def run_one(self, pdb_path, output_dir):

        # parse structure object (permissive flag but let's not lose any atoms by using a custom builder)
        parser = PDB.PDBParser(PERMISSIVE=1, QUIET=True, structure_builder=NonUniqueStructureBuilder())
        # only consider the first model in the pdb file
        structure = parser.get_structure('X', pdb_path)
        model = structure[0]

        # Get ligand (het) to extract the site around
        if self.ligand_fname_pattern[0]:
            ligand_path = pdb_path.replace(self.ligand_fname_pattern[0], self.ligand_fname_pattern[1])
            ligand = parser.get_structure('L', ligand_path)
            het_list = list(ligand.get_residues())
        else:
            # get het entries of interest (filter using static dictionaries)
            het_list = get_het_residues_from_pdb(model, remove_duplicates=False, min_lig_atoms=self.min_lig_atoms,
                                                 allowed_names=self.allowed_lig_names)

        # Setup a PDB writer and load protein
        io = PDB.PDBIO()
        io.set_structure(model)

        # create output directory and split input pdb_path
        os.makedirs(output_dir, exist_ok=True)
        name, ext = os.path.basename(pdb_path).rsplit('.', 1)

        for n, het in enumerate(het_list):

            # Set name of output site file
            if self.include_het_resname:
                site_name = f"{name}_site_{n+1}_{het.resname}.{ext}"
            else:
                site_name = f"{name}_site_{n+1}.{ext}"
            fname = os.path.join(output_dir, site_name)

            io.save(fname, NearLigandSelect(self.distance_threshold, het, keep_lig_in_site=False,
                                            keep_water=self.keep_water, keep_other_hets=self.keep_other_hets))
            if not self.ligand_fname_pattern[0]:
                if self.include_het_resname:
                    lig_name = f"{name}_lig_{n+1}_{het.resname}.{ext}"
                else:
                    lig_name = f"{name}_lig_{n+1}.{ext}"
                io.save(os.path.join(output_dir, lig_name), LigandOnlySelect(het))

        if self.save_clean_structure:
            io.save(os.path.join(output_dir, f'{name}_clean.{ext}'), ChainOnlySelect())
