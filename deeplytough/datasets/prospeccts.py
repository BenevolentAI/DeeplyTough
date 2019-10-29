import os
import glob
import pickle
import requests
import concurrent.futures
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from misc.utils import htmd_featurizer
from misc.ligand_extract import PocketFromLigandDetector

import logging
logger = logging.getLogger(__name__)


class Prospeccts:
    """ ProSPECCTs dataset by Ehrt et al (http://www.ccb.tu-dortmund.de/ag-koch/prospeccts/) """

    dbnames = ['P1', 'P1.2', 'P2', 'P3', 'P4', 'P5', 'P5.2', 'P6', 'P6.2', 'P7']
    
    def __init__(self, dbname):
        self.dbname = dbname

    @staticmethod
    def _extract_pocket_and_get_uniprot(pdbpath):
        fname = os.path.basename(pdbpath).split('.')[0]
        if '_' in fname:
            return None, None

        # 1) Extract the pocket
        detector = PocketFromLigandDetector(include_het_resname=False, save_clean_structure=True, keep_other_hets=False, min_lig_atoms=1, allowed_lig_names=['LIG'])
        detector.run_one(pdbpath, os.path.dirname(pdbpath))

        # 2) Attempt to map to Uniprots (fails from time to time, return 'None' in that case)
        pdb_code = fname[:4].lower()
        query_chain_id = fname[4].upper() if len(fname)>4 else ''
        result = set()
        try:
            r = requests.get('http://www.ebi.ac.uk/pdbe/api/mappings/{}/{}'.format('uniprot', pdb_code))
            fam = r.json()[pdb_code]['UniProt']
        except Exception as e:
            # this logically fails for artificial proteins not in PDB, such as in decoys (P3, P4), but that's fine.
            logger.warning('PDB not found {} {} {}'.format(e, pdb_code, query_chain_id))
            return fname, 'None'
        for fam_id in fam.keys():
            for chain in fam[fam_id]['mappings']:
                if not query_chain_id:
                    result.add(fam_id)
                elif chain['chain_id'] == query_chain_id:
                    if len(result) > 0 and fam_id != next(iter(result)):
                        logger.warning('Duplicate chain {} {}'.format(fam_id, result))
                    result.add(fam_id)
        if not result:
            logger.warning('Chain not found {}'.format(pdb_code))
        return fname, result
    
    def preprocess_once(self):
        logger.info('Preprocessing: extracting pockets and obtaining uniprots, this will take time.')
        all_pdbs = glob.glob(os.environ['STRUCTURE_DATA_DIR'] + '/prospeccts/**/*.pdb', recursive=True)
        code_to_uniprot = {}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for code, uniprot in executor.map(Prospeccts._extract_pocket_and_get_uniprot, all_pdbs):
                if code:
                    code_to_uniprot[code] = uniprot
        pickle.dump(code_to_uniprot, open(os.environ['STRUCTURE_DATA_DIR'] + '/prospeccts/code_to_uniprot.pickle', 'wb'))

        htmd_featurizer(self.get_structures(), skip_existing=True)

    def _prospeccts_paths(self):
        if self.dbname== 'P1':
            dir1, dir2, listfn = 'identical_structures', 'identical_structures', 'identical_structures.csv'
        elif self.dbname== 'P1.2':
            dir1, dir2, listfn = 'identical_structures_similar_ligands', 'identical_structures_similar_ligands', 'identical_structures_similar_ligands.csv'
        elif self.dbname== 'P2':
            dir1, dir2, listfn = 'NMR_structures', 'NMR_structures', 'NMR_structures.csv'
        elif self.dbname== 'P3':
            dir1, dir2, listfn = 'decoy', 'decoy_structures', 'decoy_structures5.csv'
        elif self.dbname== 'P4':
            dir1, dir2, listfn = 'decoy', 'decoy_shape_structures', 'decoy_structures5.csv'
        elif self.dbname== 'P5':
            dir1, dir2, listfn = 'kahraman_structures', 'kahraman_structures', 'kahraman_structures80.csv'
        elif self.dbname== 'P5.2':
            dir1, dir2, listfn = 'kahraman_structures', 'kahraman_structures', 'kahraman_structures.csv'
        elif self.dbname== 'P6':
            dir1, dir2, listfn = 'barelier_structures', 'barelier_structures', 'barelier_structures.csv'
        elif self.dbname== 'P6.2':
            dir1, dir2, listfn = 'barelier_structures', 'barelier_structures_cofactors', 'barelier_structures.csv'
        elif self.dbname== 'P7':
            dir1, dir2, listfn = 'review_structures', 'review_structures', 'review_structures.csv'
        else:
            raise NotImplementedError
        return dir1, dir2, listfn

    def get_structures(self):
        """ Get list of PDB structures with metainfo """
        dir1, dir2, listfn = self._prospeccts_paths()
        root = os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'prospeccts', dir1)
        npz_root = os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'processed/htmd/prospeccts', dir1)

        db_pdbs = set()
        with open(os.path.join(root, listfn)) as f:
            for line in f.readlines():
                tokens = line.split(',')
                db_pdbs.add(tokens[0])
                db_pdbs.add(tokens[1])

        try:
            code5_to_uniprot = pickle.load(open(os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'prospeccts', 'code_to_uniprot.pickle'), 'rb'))
        except FileNotFoundError:
            logger.warning('code_to_uniprot.pickle not found, please call preprocess_once()')
            code5_to_uniprot = None

        entries = []
        for pdb in db_pdbs:
            entries.append({'protein': root + '/{}/{}_clean.pdb'.format(dir2, pdb),
                     'pocket': root + '/{}/{}_site_1.pdb'.format(dir2, pdb),
                     'ligand': root + '/{}/{}_lig_1.pdb'.format(dir2, pdb),
                     'protein_htmd': npz_root + '/{}/{}_clean.npz'.format(dir2, pdb),
                     'code5': pdb,
                     'code': pdb[:4],
                     'uniprot': code5_to_uniprot[pdb] if code5_to_uniprot else 'None'
                     })

        return entries

    def evaluate_matching(self, descriptor_entries, matcher):
        """
        Evaluate pocket matching on one Prospeccts dataset. The evaluation metrics is AUC.
        :param descriptor_entries: List of entries
        :param matcher: PocketMatcher instance
        :return:
        """

        target_dict = {d['code5']: d for d in descriptor_entries}
        pairs = []
        positives = []

        dir1, dir2, listfn = self._prospeccts_paths()
        root = os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'prospeccts', dir1)

        with open(os.path.join(root, listfn)) as f:
            for line in f.readlines():
                tokens = line.split(',')
                id1, id2, cls = tokens[0], tokens[1], tokens[2].strip()
                if id1 in target_dict and id2 in target_dict:
                    pairs.append((target_dict[id1], target_dict[id2]))
                    positives.append(cls=='active')
                else:
                    logger.warning('Detection entry missing for {},{}'.format(id1, id2))

        scores = matcher.pair_match(pairs)

        goodidx = np.flatnonzero(np.isfinite(np.array(scores)))
        if len(goodidx) != len(scores):
            logger.warning('Ignoring {} pairs'.format(len(scores) - len(goodidx)))
            positives_clean, scores_clean = np.array(positives)[goodidx],  np.array(scores)[goodidx]
        else:
            positives_clean, scores_clean = positives, scores

        fpr, tpr, roc_thresholds = roc_curve(positives_clean, scores_clean)
        auc = roc_auc_score(positives_clean, scores_clean)

        results = {'auc': auc, 'fpr': fpr, 'tpr': tpr, 'th_roc': roc_thresholds,
                   'pairs': pairs, 'scores': scores, 'pos_mask': positives}
        return results

