from abc import abstractmethod


class PocketMatcher(object):
    """
    Base class for pocket matcher
    """

    @abstractmethod
    def bipartite_match(self, entries_a, entries_b):
        """
        Computes all matches between pockets from `entries_a` and pockets from `entries_b`.

        :param entries_a: List of dicts. Required keys: `protein`, `pocket`.
        :param entries_b: List of dicts. Required keys: `protein`, `pocket`.
        :return: np.array, score matrix
        """
        raise NotImplementedError

    @abstractmethod
    def pair_match(self, entry_pairs):
        """
        Computes matches between given pairs of entries.

        :param entry_pairs: List of tuples of dicts. Required keys: `protein`, `pocket`.
        :return: np.array, score vector
        """
        raise NotImplementedError

    @abstractmethod
    def complete_match(self, entries):
        """
        Computes all matches between given `entries`.

        :param entries: List of dicts. Required keys: `protein`, `pocket`.
        :return: np.array, score matrix
        """
        raise NotImplementedError
