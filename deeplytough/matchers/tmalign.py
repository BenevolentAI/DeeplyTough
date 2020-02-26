import concurrent.futures
import logging
import os
import Bio.PDB as PDB
import numpy as np
import subprocess

from misc.utils import NonUniqueStructureBuilder
from misc.ligand_extract import PocketFromLigandDetector
from .pocket_matcher import PocketMatcher

logger = logging.getLogger(__name__)


def make_full_pocket(entry):
    """ Tough: FPocket selects only atoms within the detected pocket but we need complete residues for TM Align"""
    if '_vert.pqr' not in entry['pocket']:
        return entry['pocket']
    fname = entry['pocket'].replace('_vert.pqr', '_atm_full.pdb')
    if not os.path.exists(fname):
        parser = PDB.PDBParser(PERMISSIVE=1, QUIET=True, structure_builder=NonUniqueStructureBuilder())
        structure = parser.get_structure('X', entry['protein'])
        model = structure[0]

        class ResidueSelect(PDB.Select):
            def __init__(self, structure):
                self.residues = []
                for chains in structure:
                    for chain in chains:
                        for residue in chain:
                            self.residues.append(residue.id[1])
                self.residues = set(self.residues)

            def accept_residue(self, residue):
                return residue.id[1] in self.residues

        io = PDB.PDBIO()
        io.set_structure(model)
        io.save(fname, ResidueSelect(parser.get_structure('X', entry['pocket'].replace('_vert.pqr', '_atm.pdb'))))
    return fname


def make_tough_lig_pocket(entry):
    """ Tough: Extract rather the pocket around ligand; TMAlign does not seem to perform well for detected pockets."""
    if '_vert.pqr' not in entry['pocket']:
        return entry['pocket']
    fname = entry['protein'].replace('.pdb', '_site_1.pdb')
    if not os.path.exists(fname):
        detector = PocketFromLigandDetector(include_het_resname=False, save_clean_structure=False, keep_other_hets=False, min_lig_atoms=3, ligand_fname_pattern=('.pdb','00.pdb'))
        detector.run_one(entry['protein'], os.path.dirname(entry['protein']))
    return fname


def align_worker(tr, te):
    command = ['/home/msimonovsky/DeeplyTough/tm/TMalign', te['pocket'], tr['pocket']]
    output = subprocess.check_output(command, timeout=60, universal_newlines=True)
    scores = []
    for line in output.split('\n'):
        if 'TM-score=' in line:
            scores.append(float(line.split()[1]))
    if len(scores) == 0:
        return np.nan, tr
    score = max(scores)
    return score, tr

def _align_worker_unpack(args):
    return align_worker(*args)


class TMAlign(PocketMatcher):
    """
    Pocket matching with TMAlign
    """

    def __init__(self, nworkers=None, tough_lig_pockets=True):
        """
        """
        self.nworkers = nworkers if nworkers > 1 else None
        self.tough_lig_pockets = tough_lig_pockets

    def pair_match(self, entry_pairs):
        for e1, e2 in entry_pairs:
            if not self.tough_lig_pockets:
                e1['pocket'] = make_full_pocket(e1)
                e2['pocket'] = make_full_pocket(e2)
            else:
                e1['pocket'] = make_tough_lig_pocket(e1)
                e2['pocket'] = make_tough_lig_pocket(e2)

        scores = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.nworkers) as executor:
            for score, tr in executor.map(_align_worker_unpack, entry_pairs):
                assert tr['pocket'] == entry_pairs[len(scores)][0]['pocket'], 'not in order map, should not happen'
                scores.append(score)
        return scores        

    def complete_match(self, entries):
        product = [(e1, e2) for e1 in entries for e2 in entries]
        scores = self.pair_match(product)
        return np.array(scores).reshape((len(entries), len(entries)))
