import os
import subprocess
import Bio.PDB as PDB
import pickle
import requests
import concurrent.futures
import tempfile
import urllib.request
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, GroupShuffleSplit
from misc.utils import htmd_featurizer, voc_ap

import logging
logger = logging.getLogger(__name__)


class ToughM1:
    """ TOUGH-M1 dataset by Govindaraj and Brylinski(https://osf.io/6ngbs/wiki/home/) """

    def __init__(self):
        pass

    @staticmethod
    def _preprocess_worker(entry):

        def struct_to_centroid(structure):
            return np.mean(np.array([atom.get_coord() for atom in structure.get_atoms()]), axis=0)

        def pdb_chain_to_uniprot(pdb_code, query_chain_id):
            result = 'None'
            r = requests.get('http://www.ebi.ac.uk/pdbe/api/mappings/{}/{}'.format('uniprot', pdb_code))
            fam = r.json()[pdb_code]['UniProt']

            for fam_id in fam.keys():
                for chain in fam[fam_id]['mappings']:
                    if chain['chain_id'] == query_chain_id:
                        if result != 'None' and fam_id != result:
                            logger.warning('DUPLICATE %s %s', fam_id, result)
                        result = fam_id
            if result == 'None':
                logger.warning('NOT FOUND %s', pdb_code)
            return result

        try:
            # 1) We won't be using provided `.fpocket` files because they don't contain the actual atoms, just
            # Voronoii centers. So we run fpocket2 ourselves, it seems to be equivalent to published results.
            try:
                command = ['fpocket2', '-f', entry['protein']]
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                logger.warning('Calling fpocket2 failed, please make sure it is on the PATH')
                raise e

            # 2) TOUGH authors have unfortunately sometimes renamed chains so one cannot directly get the uniprot
            # corresponding to a given chain. So we need to first locate the corresponding chains in the original
            # pdb files, get their ids and translate those to uniprot with a webservice.
            parser = PDB.PDBParser(PERMISSIVE=True, QUIET=True)
            tough_str = parser.get_structure('t', entry['protein'])
            tough_c = struct_to_centroid(tough_str)

            with tempfile.TemporaryDirectory() as tmpdir:
                fn = tmpdir + '/prot.pdb'
                urllib.request.urlretrieve('http://files.rcsb.org/download/{}.pdb'.format(entry['code'].upper()), fn)
                orig_str = parser.get_structure('o', fn)

            # TOUGH authors haven't recentered the chains so we can roughly find them just by centroids:)
            dists = []
            ids = []
            for model in orig_str:
                for chain in model:
                    dists.append(np.linalg.norm(struct_to_centroid(chain) - tough_c))
                    ids.append(chain.id)
            chain_id = ids[np.argmin(dists)]
            if np.min(dists) > 5:
                logger.warning('SUSPICIOUS DIST %f %f %s', dists, ids, entry['code'])

            uniprot = pdb_chain_to_uniprot(entry['code'], chain_id)
            return [entry['code5'], uniprot, chain_id]
        except Exception as e:
            print(e)
            return [entry['code5'], 'None', 'X']

    def preprocess_once(self):
        """ Re-run fpocket2 and try to obtain UniprotID for each PDB entry. Needs to be called just once in a lifetime. """

        code5_to_uniprot = {}
        code5_to_chainid = {}
        uniprot_to_code5 = defaultdict(list)

        logger.info('Preprocessing: obtaining uniprots, this will take time.')
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for code5, uniprot, chain_id in executor.map(ToughM1._preprocess_worker, self.get_structures()):
                code5_to_uniprot[code5] = uniprot
                code5_to_chainid[code5] = chain_id
                uniprot_to_code5[uniprot] = uniprot_to_code5[uniprot] + [code5]

        pickle.dump({'code5_to_uniprot': code5_to_uniprot, 'uniprot_to_code5': uniprot_to_code5, 'code5_to_chainid': code5_to_chainid},
                    open(os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'TOUGH-M1', 'code_uniprot_translation.pickle'), 'wb'))

        htmd_featurizer(self.get_structures(), skip_existing=True)

    def get_structures(self):
        """ Get list of PDB structures with metainfo """

        root = os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'TOUGH-M1', 'TOUGH-M1_dataset')
        npz_root = os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'processed/htmd/TOUGH-M1/TOUGH-M1_dataset')
        try:
            code5_to_uniprot = pickle.load(open(os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'TOUGH-M1', 'code_uniprot_translation.pickle'), 'rb'))['code5_to_uniprot']
        except FileNotFoundError:
            logger.warning('code_uniprot_translation.pickle not found, please call preprocess_once()')
            code5_to_uniprot = None

        entries = []
        with open(os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'TOUGH-M1', 'TOUGH-M1_pocket.list')) as f:
            for line in f.readlines():
                code5, pocketnr, _ = line.split()
                entries.append({'protein': root + '/{0}/{0}.pdb'.format(code5),
                                'pocket': root + '/{0}/{0}_out/pockets/pocket{1}_vert.pqr'.format(code5, int(pocketnr) - 1),
                                'ligand': root + '/{0}/{0}00.pdb'.format(code5),
                                'protein_htmd': npz_root + '/{0}/{0}.npz'.format(code5),
                                'code5': code5,
                                'code': code5[:4],
                                'uniprot': code5_to_uniprot[code5] if code5_to_uniprot else 'None'
                                })

        return entries

    def get_structures_splits(self, fold_nr, strategy='uniprot_folds', n_folds=5, seed=0):
        pdb_entries = self.get_structures()

        if strategy == 'pdb_folds':
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            folds = list(splitter.split(pdb_entries))
            train_idx, test_idx = folds[fold_nr]
            return [pdb_entries[i] for i in train_idx], [pdb_entries[i] for i in test_idx]

        elif strategy == 'uniprot_folds':
            splitter = GroupShuffleSplit(n_splits=n_folds, test_size=1.0/n_folds, random_state=seed)
            pdb_entries = list(filter(lambda entry: entry['uniprot'] != 'None', pdb_entries))
            folds = list(splitter.split(pdb_entries, groups=[e['uniprot'] for e in pdb_entries]))
            train_idx, test_idx = folds[fold_nr]
            return [pdb_entries[i] for i in train_idx], [pdb_entries[i] for i in test_idx]

        elif strategy == 'none':
            return pdb_entries, pdb_entries
        else:
            raise NotImplementedError

    def evaluate_matching(self, descriptor_entries, matcher):
        """
        Evaluate pocket matching on TOUGH-M1 dataset. The evaluation metrics is AUC.
        :param descriptor_entries: List of entries
        :param matcher: PocketMatcher instance
        :return:
        """

        target_dict = {d['code5']: d for d in descriptor_entries}
        pairs = []
        positives = []

        def parse_file_list(f):
            f_pairs = []
            for line in f.readlines():
                id1, id2 = line.split()[:2]
                if id1 in target_dict and id2 in target_dict:
                    f_pairs.append((target_dict[id1], target_dict[id2]))
            return f_pairs

        with open(os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'TOUGH-M1', 'TOUGH-M1_positive.list')) as f:
            pos_pairs = parse_file_list(f)
            pairs.extend(pos_pairs)
            positives.extend([True] * len(pos_pairs))

        with open(os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'TOUGH-M1', 'TOUGH-M1_negative.list')) as f:
            neg_pairs = parse_file_list(f)
            pairs.extend(neg_pairs)
            positives.extend([False] * len(neg_pairs))

        scores = matcher.pair_match(pairs)

        goodidx = np.flatnonzero(np.isfinite(np.array(scores)))
        if len(goodidx) != len(scores):
            logger.warning('Ignoring {} pairs'.format(len(scores) - len(goodidx)))
            positives_clean, scores_clean = np.array(positives)[goodidx],  np.array(scores)[goodidx]
        else:
            positives_clean, scores_clean = positives, scores

        fpr, tpr, roc_thresholds = roc_curve(positives_clean, scores_clean)
        auc = roc_auc_score(positives_clean, scores_clean)
        precision, recall, thresholds = precision_recall_curve(positives_clean, scores_clean)
        ap = voc_ap(recall[::-1], precision[::-1])

        results = {'ap': ap, 'pr': precision, 're': recall, 'th': thresholds,
                   'auc': auc, 'fpr': fpr, 'tpr': tpr, 'th_roc': roc_thresholds,
                   'pairs': pairs, 'scores': scores, 'pos_mask': positives}
        return results
