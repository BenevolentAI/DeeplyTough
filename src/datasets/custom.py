import os
from misc.utils import htmd_featurizer


class Custom:
    """ An arbitrary user dataset

    Assumes that the dataset is placed in `$STRUCTURE_DATA_DIR/relpath`, containing
    bunch of protein and pocket structures, which are referred in `pairs.csv`. This
    file contains a quadruplet on each line indicating matches to evaluate:

    relative_path_to_pdbA, relative_path_to_pocketA, relative_path_to_pdbB, relative_path_to_pocketB
    """

    def __init__(self, relpath='custom'):
        self.relpath = relpath

    def preprocess_once(self):
        """ Computes featurization """
        htmd_featurizer(self.get_structures(), skip_existing=True)

    def get_structures(self):
        """ Get list of PDB structures with metainfo """

        root = os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), self.relpath)
        npz_root = os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'processed/htmd', self.relpath)

        custom_pdbs = set()
        with open(os.path.join(root, 'pairs.csv')) as f:
            for i, line in enumerate(f.readlines()):
                tokens = line.split(',')
                assert len(tokens)==4, 'pairs.csv is expected to have four columns.'
                custom_pdbs.add((tokens[0].strip(), tokens[1].strip()))
                custom_pdbs.add((tokens[2].strip(), tokens[3].strip()))

        entries = []
        for pdb, pocket in custom_pdbs:
            pdb1 = pdb if os.path.splitext(pdb)[1] != '' else pdb + '.pdb'
            pocket1 = pocket if os.path.splitext(pocket)[1] != '' else pocket + '.pdb'
            entries.append({'protein': os.path.join(root, pdb1),
                     'pocket': os.path.join(root, pocket1),
                     'protein_htmd': os.path.join(npz_root, pdb1.replace('.pdb', '.npz')),
                     'key': pdb + ',' + pocket})

        return entries

    def evaluate_matching(self, descriptor_entries, matcher):
        """
        Compute pocket matching scores on the custom dataset.
        :param descriptor_entries: List of entries
        :param matcher: PocketMatcher instance
        :return:
        """

        target_dict = {d['key']: d for d in descriptor_entries}
        root = os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), self.relpath)

        pairs = []
        with open(os.path.join(root, 'pairs.csv')) as f:
            for i, line in enumerate(f.readlines()):
                tokens = line.split(',')
                assert len(tokens)==4, 'pairs.csv is expected to have four columns.'
                key1 = tokens[0].strip() + ',' + tokens[1].strip()
                key2 = tokens[2].strip() + ',' + tokens[3].strip()
                pairs.append((target_dict[key1], target_dict[key2]))

        scores = matcher.pair_match(pairs)
        return {'pairs': pairs, 'scores': scores}
