import numpy as np
import os
from .pocket_matcher import PocketMatcher


class ToughOfficials(PocketMatcher):
    """
    Return the precomputed results for several methods in official Tough dataset repo
    """

    def __init__(self, alg_name, score_column):
        self.scores = {}
        for cls in ['positive', 'negative']:
            with open(os.path.join(
                    os.environ.get('STRUCTURE_DATA_DIR'), 'TOUGH-M1', f'{alg_name}-TOUGH-M1_{cls}.score')) as f:
                for line in f.readlines():
                    s = line.split()
                    self.scores[s[0] + s[1]] = float(s[score_column])

    def pair_match(self, entry_pairs):

        scores = np.full((len(entry_pairs)), np.nan)

        for i, (entry_a, entry_b) in enumerate(entry_pairs):
            score = self.scores.get(entry_a['code5'] + entry_b['code5'], None)
            if score is None:
                score = self.scores.get(entry_a['code5'] + entry_b['code5'], None)
            assert score is not None
            scores[i] = score

        return scores


