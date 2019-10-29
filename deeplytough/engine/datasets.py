import os
import math
import numpy as np
import random
from torch.utils.data import Dataset
import transforms3d
import ctypes
import htmd.home
from datasets import ToughM1, Vertex, Prospeccts
from misc.utils import center_from_pdb_file

import logging
logger = logging.getLogger(__name__)

occupancylib = ctypes.cdll.LoadLibrary(os.path.join(htmd.home.home(libDir=True), "occupancy_ext.so"))


class VoxelizedDataset(Dataset):
    """ Abstract base class for dataset of voxelized proteins. """

    # ‘hydrophobic’, ‘aromatic’, ‘hbond_acceptor’, ‘hbond_donor’, ‘positive_ionizable’, ‘negative_ionizable’, ‘metal’, ‘occupancies’
    num_channels = 8

    def __init__(self, pdb_list, box_size, augm_rot=False, augm_mirror_prob=0.0):
        """
        :param pdb_list: List of pdb files (as dicts, with key 'protein_htmd').
        :param box_size: Patch size
        :param augm_rot: Rotation augmentation
        :param augm_mirror_prob: Mirroring probability for each axis
        """

        self.pdb_list = []
        self.pdb_idx = []

        for i, pdb_entry in enumerate(pdb_list):
            if not os.path.exists(pdb_entry['protein_htmd']):
                logging.warning('HTMD featurization file not found: %s, corresponding pdb likely could not have been parsed', pdb_entry['protein_htmd'])
                continue
            self.pdb_list.append(pdb_entry)
            self.pdb_idx.append(i)

        assert len(self.pdb_list) > 0, 'No HTMD could be found but {} PDB files were given, please call preprocess_once() on the dataset'.format(len(pdb_list))
        logging.info('Dataset size: %d', len(self.pdb_list))

        self._resolution = 1.0
        self._box_size = box_size
        self._augm_rot = augm_rot
        self._augm_mirror_prob = augm_mirror_prob

    def __len__(self):
        return len(self.pdb_list)

    def __getitem__(self, idx):
        raise NotImplementedError()

    def _sample_augmentation(self):
        """
        Samples random rotation and mirroring, returns a 3x3 matrix
        """
        M = np.eye(3)
        if self._augm_rot:
            angle = random.uniform(0, 2*math.pi)
            M = np.dot(transforms3d.axangles.axangle2mat(np.random.uniform(size=3), angle), M)
        if self._augm_mirror_prob > 0:
            if random.random() < self._augm_mirror_prob/2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
            if random.random() < self._augm_mirror_prob/2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,1,0]), M)
            if random.random() < self._augm_mirror_prob/2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), M)
        return M

    def _extract_volume(self, coords, channels, center, num_voxels, resolution=1.0):
        """
        Computes dense volume for htmd preprocessed coordinates
        """
        assert center.size == 3
        num_voxels = np.array(num_voxels)
        if num_voxels[0] % 2 == 0 and num_voxels[1] % 2 == 0 and num_voxels[2] % 2 == 0:
            # place the center point at one of the two middle voxels (not centered, but center will not be quantized)
            start = center - resolution * (num_voxels // 2)
            end = center + resolution * (num_voxels // 2 - 1)
        else:
            # center the box around the center point
            start = center - resolution * (num_voxels // 2)
            end = center + resolution * (num_voxels // 2)

        gridx, gridy, gridz = np.meshgrid(np.linspace(start[0], end[0], num_voxels[0]),
                                          np.linspace(start[1], end[1], num_voxels[1]),
                                          np.linspace(start[2], end[2], num_voxels[2]), indexing='ij')

        centers = np.stack([gridx, gridy, gridz], axis=-1).reshape(-1,3)
        volume = self._getOccupancyC(coords, centers, channels)
        volume = volume.reshape(num_voxels[0], num_voxels[1], num_voxels[2], -1).transpose((3,0,1,2)).astype(np.float32)
        return volume, start, centers

    @staticmethod
    def _getOccupancyC(coords, centers, channelsigmas):  # adapted from voxeldescriptors.py in HTMD
        """ Calls the C code to calculate the voxels values for each property."""
        centers = centers.astype(np.float64)
        coords = coords.astype(np.float32)
        channelsigmas = channelsigmas.astype(np.float64)

        nchannels = channelsigmas.shape[1]
        occus = np.zeros((centers.shape[0], nchannels), dtype=np.float64)

        occupancylib.descriptor_ext(centers.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           channelsigmas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           occus.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           ctypes.c_int(occus.shape[0]),  # n of centers
                           ctypes.c_int(coords.shape[0]),  # n of atoms
                           ctypes.c_int(nchannels))  # n of channels
        return occus


class PdbTupleVoxelizedDataset(VoxelizedDataset):
    """ Abstract base class for dataset of tuples of subvolumes of voxelized proteins. """

    def __init__(self, pos_pairs, neg_pairs, pdb_list, box_size, augm_rot=False, augm_mirror_prob=0.0, max_sampling_dist=4.0, augm_robustness=False, augm_decoy_prob=0):
        super().__init__(pdb_list, box_size, augm_rot, augm_mirror_prob)

        self._max_sampling_dist = max_sampling_dist
        self._augm_robustness = augm_robustness
        self._decoy_prob = augm_decoy_prob

        # map pdb code -> entry
        self._pdb_map = {}
        for i, pdb_entry in enumerate(self.pdb_list):
            code = pdb_entry['code5'] if 'code5' in pdb_entry else pdb_entry['code']
            self._pdb_map[code] = i

        # filter pairs to those supported by pdbs
        self._pos_pairs = list(filter(lambda p: p[0] in self._pdb_map and p[1] in self._pdb_map, pos_pairs))
        self._neg_pairs = list(filter(lambda p: p[0] in self._pdb_map and p[1] in self._pdb_map, neg_pairs))
        logging.info('Dataset positive pairs: %d, negative pairs: %d', len(self._pos_pairs), len(self._neg_pairs))
        num_eff_pdbs = set([p[0] for p in self._pos_pairs] + [p[1] for p in self._pos_pairs] + [p[0] for p in self._neg_pairs] + [p[1] for p in self._neg_pairs])
        logging.info('Effective number of PDB files: %d', len(num_eff_pdbs))
        assert len(self._pos_pairs) > 0 and len(self._neg_pairs) > 0

    def _get_patch(self, idx, allow_decoy=False):

        container = np.load(self.pdb_list[idx]['protein_htmd'])
        struct_coords = container['coords']
        struct_channels = container['channels']
        center = center_from_pdb_file(self.pdb_list[idx]['pocket'])

        v, r = np.random.normal(size=3), np.random.uniform(size=1, high=self._max_sampling_dist)
        center = center + r * v / (np.linalg.norm(v) + 1e-10)
        centers = [center]

        # Decoys (random negative points) can be added to increase the variability of negatives (e.g. empty space:P)
        # Should be added into only one member of a negative pair (then matching anything to a decoy will be penalized;
        # having decoys in both pair members would be difficult, a decoy might in fact match another decoy)
        if allow_decoy and random.uniform(0, 1) <= self._decoy_prob:
            struct_min = np.amin(struct_coords, axis=0)
            struct_max = np.amax(struct_coords, axis=0)
            centers = [struct_min + (struct_max - struct_min) * np.random.uniform(size=3)]

        shape = [self._box_size]*3
        volumes = []

        if self._augm_robustness:
            centers.extend(centers)  # same centers but with different augmentation

        for center in centers:
            # data augmentation by rotation and mirroring
            if self._augm_rot or self._augm_mirror_prob > 0:
                M = self._sample_augmentation()
                struct_coords_aug = np.dot(struct_coords, M.T)
                center = np.dot(center, M.T)
            else:
                struct_coords_aug = struct_coords

            # crop point cloud and convert it into a volume
            volume, start, grid_pts = self._extract_volume(struct_coords_aug, struct_channels, center, shape, self._resolution)
            volumes.append(volume)

        return volumes


class PdbPairVoxelizedDataset(PdbTupleVoxelizedDataset):
    """ Dataset of pairs of voxelized pockets. """

    def __len__(self):
        # positive pairs as the driving entity
        return len(self._pos_pairs) * 2

    def __getitem__(self, idx):
        if idx % 2 == 0:
            cls = 1  # positive class
            pair = self._pos_pairs[idx // 2]
        else:
            cls = 0  # negative class
            pair = random.choice(self._neg_pairs)

        first_vols = self._get_patch(self._pdb_map[pair[0]])
        second_vols = self._get_patch(self._pdb_map[pair[1]], allow_decoy=(cls=='neg'))

        return {'inputs': np.stack(first_vols + second_vols), 'targets': np.array([cls], dtype=np.float32)}


class PointOfInterestVoxelizedDataset(VoxelizedDataset):
    """ Dataset of voxelized subvolumes around interest points. """

    def __init__(self, pdb_list, point_list, box_size):
        super().__init__(pdb_list, box_size=box_size, augm_rot=False, augm_mirror_prob=0)
        self._extraction_points = point_list

    def __getitem__(self, idx):
        container = np.load(self.pdb_list[idx]['protein_htmd'])
        struct_coords = container['coords']
        struct_channels = container['channels']
        shape = [self._box_size]*3
        volumes = []

        for center in self._extraction_points[self.pdb_idx[idx]]:
            volume, start, grid_pts = self._extract_volume(struct_coords, struct_channels, center, shape, self._resolution)
            volumes.append(volume)

        return {'inputs': np.stack(volumes), 'pdb_idx': self.pdb_idx[idx]}


def create_tough_dataset(args, fold_nr, n_folds=5, seed=0, exclude_Vertex_from_train=False, exclude_Prospeccts_from_train=False):

    if args.db_preprocessing:
        ToughM1().preprocess_once()
        Vertex().preprocess_once()
        ToughM1().preprocess_once()
        for dbname in Prospeccts.dbnames: Prospeccts(dbname).preprocess_once()

    pdb_train, pdb_test = ToughM1().get_structures_splits(fold_nr, strategy=args.db_split_strategy, n_folds=n_folds, seed=seed)

    if exclude_Vertex_from_train:
        vertex = Vertex().get_structures()
        logger.info('Before Vertex filter {}'.format(len(pdb_train)))
        if exclude_Vertex_from_train == 'uniprot':
            vertex_ups = set([entry['uniprot'] for entry in vertex] + ['None'])
            pdb_train = list(filter(lambda entry: entry['uniprot'] not in vertex_ups, pdb_train))
        elif exclude_Vertex_from_train == 'pdb':
            vertex_pdbs = set([entry['code'] for entry in vertex])
            pdb_train = list(filter(lambda entry: entry['code'] not in vertex_pdbs, pdb_train))
        else:
            raise NotImplementedError()
        logger.info('After Vertex filter {}'.format(len(pdb_train)))

    if exclude_Prospeccts_from_train:
        all_prospeccts = [entry for dbname in Prospeccts.dbnames for entry in Prospeccts(dbname).get_structures()]
        logger.info('Before Prospeccts filter {}'.format(len(pdb_train)))
        if exclude_Prospeccts_from_train == 'uniprot':
            prospeccts_ups = set([u for entry in all_prospeccts for u in entry['uniprot']] + ['None'])
            pdb_train = list(filter(lambda entry: entry['uniprot'] not in prospeccts_ups, pdb_train))
        elif exclude_Prospeccts_from_train == 'pdb':
            prospeccts_pdbs = set([entry['code'] for entry in all_prospeccts])
            pdb_train = list(filter(lambda entry: entry['code'].lower() not in prospeccts_pdbs, pdb_train))
        else:
            raise NotImplementedError()
        logger.info('After Prospeccts filter {}'.format(len(pdb_train)))

    with open(os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'TOUGH-M1', 'TOUGH-M1_positive.list')) as f:
        pos_pairs = [line.split()[:2] for line in f.readlines()]
    with open(os.path.join(os.environ.get('STRUCTURE_DATA_DIR'), 'TOUGH-M1', 'TOUGH-M1_negative.list')) as f:
        neg_pairs = [line.split()[:2] for line in f.readlines()]

    rndstate = random.getstate()
    random.seed(seed)
    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)
    random.setstate(rndstate)

    train_db = PdbPairVoxelizedDataset(pos_pairs, neg_pairs, pdb_train, box_size=args.patch_size, augm_rot=args.augm_rot, augm_mirror_prob=args.augm_mirror_prob,
                        max_sampling_dist=args.augm_sampling_dist, augm_robustness=args.stability_loss_weight>0, augm_decoy_prob=args.augm_decoy_prob)
    test_db = PdbPairVoxelizedDataset(pos_pairs, neg_pairs, pdb_test, box_size=args.patch_size, augm_rot=False, augm_mirror_prob=0.0,
                       max_sampling_dist=args.augm_sampling_dist)
    return train_db, test_db
