import concurrent.futures
import logging
import os
import pickle
import string
import urllib.request
from collections import defaultdict

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from tqdm.autonotebook import tqdm

from misc.ligand_extract import PocketFromLigandDetector
from misc.utils import htmd_featurizer, voc_ap, RcsbPdbClusters

logger = logging.getLogger(__name__)


class Vertex:
    """
    Vertex dataset by Chen et al
    http://pubs.acs.org/doi/suppl/10.1021/acs.jcim.6b00118/suppl_file/ci6b00118_si_002.zip
    """

    @staticmethod
    def _download_pdb_and_extract_pocket(entry):
        code = entry['code']
        entry_dir = os.path.dirname(entry['protein'])
        os.makedirs(entry_dir, exist_ok=True)
        fname = f'{entry_dir}/{code}.pdb'
        try:
            urllib.request.urlretrieve(f'http://files.rcsb.org/download/{code.upper()}.pdb', fname)
            detector = PocketFromLigandDetector(include_het_resname=False, save_clean_structure=True,
                                                keep_other_hets=False, min_lig_atoms=3)
            detector.run_one(fname, entry_dir)
        except Exception as e:
            logger.warning(f'PROBLEM DOWNLOADING AND EXTRACTING {code}:')
            logger.exception(e)
        return code

    def preprocess_once(self):
        """
        Download pdb files and extract pocket around ligands
        """
        logger.info('Preprocessing: downloading data and extracting pockets, this will take time.')
        entries = self.get_structures(extra_mappings=False)
        
        code5_to_seqclusts = {}
        clusterer = RcsbPdbClusters(identity=30)        
        for entry in entries:
            # entries are defined by site integers in the vertex set
            chains = string.ascii_uppercase  # play it safe and take all possible chains for a protein
            seqclusts = set([clusterer.get_seqclust(entry['code'], c) for c in chains])
            code5_to_seqclusts[entry['code5']] = seqclusts
        pickle.dump({'code5_to_seqclusts': code5_to_seqclusts},
                    open(os.path.join(os.environ['STRUCTURE_DATA_DIR'], 'Vertex' , 'code5_to_seqclusts.pickle'), 'wb'))        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for _ in executor.map(Vertex._download_pdb_and_extract_pocket, entries):
                pass

        htmd_featurizer(entries, skip_existing=True)

    def get_structures(self, extra_mappings=True):
        """
        Get list of PDB structures with metainfo
        """

        root = os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'Vertex')
        npz_root = os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'processed/htmd/Vertex')

        # Read in a set of (pdb_chain, uniprot, ligand_cc) tuples
        vertex_pdbs = set()
        with open(os.path.join(root, 'protein_pairs.tsv')) as f:
            for i, line in enumerate(f.readlines()):
                if i > 1:
                    tokens = line.split('\t')
                    vertex_pdbs.add((tokens[0].lower(), tokens[2], tokens[1]))
                    vertex_pdbs.add((tokens[5].lower(), tokens[7], tokens[6]))

        code5_to_seqclusts = None        
        if extra_mappings:
            mapping = pickle.load(open(os.path.join(os.environ['STRUCTURE_DATA_DIR'], 'Vertex', 'code5_to_seqclusts.pickle'), 'rb'))
            code5_to_seqclusts = mapping['code5_to_seqclusts']

        # Generate entries for the Vertex set
        entries = []
        for n, (code5, uniprot, ligand_cc) in enumerate(vertex_pdbs):
            pdb_code = code5[:4]
            entries.append({
                'protein': root + f'/{pdb_code}/{pdb_code}_clean.pdb',
                'pocket': root + f'/{pdb_code}/{pdb_code}_site_{int(code5[5])}.pdb',
                'ligand': root + f'/{pdb_code}/{pdb_code}_lig_{int(code5[5])}.pdb',
                'protein_htmd': npz_root + f'/{pdb_code}/{pdb_code}_clean.npz',
                'code5': code5,
                'code': code5[:4],
                'lig_cc': ligand_cc,
                'uniprot': uniprot,
                'seqclusts': code5_to_seqclusts[code5] if code5_to_seqclusts else 'None'
            })
        return entries

    @staticmethod
    def evaluate_matching(descriptor_entries, matcher):
        """
        Evaluate pocket matching on Vertex dataset
        The evaluation metric is AUC

        :param descriptor_entries: List of entries
        :param matcher: PocketMatcher instance
        """

        target_dict = {d['code5']: i for i, d in enumerate(descriptor_entries)}
        prot_pairs = defaultdict(list)
        prot_positives = {}

        # Assemble dictionary pair-of-uniprots -> list_of_pairs_of_indices_into_descriptor_entries
        with open(os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'Vertex', 'protein_pairs.tsv')) as f:
            for i, line in enumerate(f.readlines()):
                if i > 1:
                    tokens = line.split('\t')
                    pdb1, id1, pdb2, id2, cls = tokens[0].lower(), tokens[2], tokens[5].lower(), tokens[7], int(tokens[-1])
                    if pdb1 in target_dict and pdb2 in target_dict:
                        key = (id1, id2) if id1 < id2 else (id2, id1)
                        prot_pairs[key] = prot_pairs[key] + [(target_dict[pdb1], target_dict[pdb2])]
                        if key in prot_positives:
                            assert prot_positives[key] == (cls == 1)
                        else:
                            prot_positives[key] = (cls == 1)

        positives = []
        scores = []
        keys_out = []

        # Evaluate each protein pairs (taking max over all pdb pocket scores, see Fig 1B in Chen et al)
        for key, pdb_pairs in tqdm(prot_pairs.items()):
            unique_idxs = list(set([p[0] for p in pdb_pairs] + [p[1] for p in pdb_pairs]))

            complete_scores = matcher.complete_match([descriptor_entries[i] for i in unique_idxs])

            sel_scores = []
            for pair in pdb_pairs:
                i, j = unique_idxs.index(pair[0]), unique_idxs.index(pair[1])
                if np.isfinite(complete_scores[i, j]):
                    sel_scores.append(complete_scores[i, j])

            if len(sel_scores) > 0:
                positives.append(prot_positives[key])
                keys_out.append(key)
                scores.append(max(sel_scores))
            else:
                logger.warning(f'Skipping a pair, could not be evaluated')

        # Calculate metrics
        fpr, tpr, roc_thresholds = roc_curve(positives, scores)
        auc = roc_auc_score(positives, scores)
        precision, recall, thresholds = precision_recall_curve(positives, scores)
        ap = voc_ap(recall[::-1], precision[::-1])

        results = {
            'ap': ap,
            'pr': precision,
            're': recall,
            'th': thresholds,
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'th_roc': roc_thresholds,
            'pairs': keys_out,
            'scores': scores,
            'pos_mask': positives
        }
        return results
