import concurrent.futures
import logging
import os
from tqdm.autonotebook import tqdm
from .pocket_matcher import PocketMatcher
from engine.predictor import load_model, match_precomputed_point_pairs, load_and_precompute_point_feats, match_precomputed_points_bipartite
from misc.utils import center_from_pdb_file

logger = logging.getLogger(__name__)


class DeeplyTough(PocketMatcher):
    """
    Pocket matching with DeeplyTough, with precomputing of descriptors.
    """

    def __init__(self, model_dir, device='cpu', batch_size=30, nworkers=1):
        """
        """
        self.model_dir = model_dir
        self.device = device
        self.batch_size = batch_size
        self.nworkers = nworkers
        self.model, self.args = load_model(model_dir, device)

    def precompute_descriptors(self, entries):
        """
        Precompute pocket descriptors/features.

        :param entries: List of entries. Required keys: `protein`, `pocket`.
        :return: entries but with `descriptor` keys.
        """

        pdb_list, point_list = [], []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            jobs = [executor.submit(center_from_pdb_file, entry['pocket']) for entry in entries]
            for job, entry in tqdm(zip(jobs, entries)):
                center = job.result()
                if center is not None:
                    pdb_list.append(entry)
                    point_list.append([center])
                else:
                    logger.warning('Pocket not found, skipping: ' + os.path.basename(entry['pocket']))

        feats = load_and_precompute_point_feats(self.model, self.args, pdb_list, point_list, self.device, self.nworkers, self.batch_size)
        for entry, feat in zip(pdb_list, feats):
            entry['descriptor'] = feat
        return pdb_list

    def pair_match(self, entry_pairs):
        """
        Computes matches between given pairs of pockets.

        :param entry_pairs: List of tuples. Required keys: `descriptors`.
        :return: np.array, score vector (negative distance)
        """

        featslist_A = [entry['descriptor'] for entry, _ in entry_pairs]
        featslist_B = [entry['descriptor'] for _, entry in entry_pairs]
        distances = match_precomputed_point_pairs(featslist_A, featslist_B)
        return -distances

    def complete_match(self, entries):

        featslist = [entry['descriptor'] for entry in entries]
        distances = match_precomputed_points_bipartite(featslist, None)
        return -distances

    def bipartite_match(self, entries_a, entries_b):

        featslist_A = [entry['descriptor'] for entry in entries_a]
        featslist_B = [entry['descriptor'] for entry in entries_b]
        distances = match_precomputed_points_bipartite(featslist_A, featslist_B)
        return -distances
