import logging
import os

import numpy as np
import torch
import torch.nn.functional as nnf
from tqdm.autonotebook import tqdm

from engine.datasets import PointOfInterestVoxelizedDataset
from engine.models import create_model

logger = logging.getLogger(__name__)


def load_model(model_dir, device):
    """
    Loads the model from file
    """
    if isinstance(device, str):
        device = torch.device(device)
    fname = os.path.join(model_dir, 'model.pth.tar') if 'pth.tar' not in model_dir else model_dir
    checkpoint = torch.load(fname, map_location=str(device))
    model = create_model(checkpoint['args'], PointOfInterestVoxelizedDataset, device)
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['args']


def load_and_precompute_point_feats(model, args, pdb_list, point_list, device, nworkers, batch_size):
    """
    Compute descriptors for every (pdb, point) pair given
    """
    model.eval()
    if isinstance(device, str):
        device = torch.device(device)

    dataset = PointOfInterestVoxelizedDataset(pdb_list, point_list, box_size=args.patch_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=nworkers)

    with torch.no_grad():
        feats = [None] * len(point_list)

        for batch in tqdm(loader):
            inputs = batch['inputs'].squeeze(1).to(device)
            outputs = model(inputs)
            if args.l2_normed_descriptors:
                outputs = nnf.normalize(outputs)
            descriptors = outputs.cpu().float()
            for b in range(descriptors.shape[0]):
                feats[batch['pdb_idx'][b]] = descriptors[b].view(-1, descriptors[b].shape[0])

        return feats


def match_precomputed_point_pairs(descriptors_A, descriptors_B):
    """
    Match pairs of descriptors. Some may be None, then their distance is NaN
    """
    with torch.no_grad():
        distances = []

        for feats_A, feats_B in tqdm(zip(descriptors_A, descriptors_B)):
            if feats_A is None or feats_B is None:
                distances.append(np.nan)
            else:
                distances.append(nnf.pairwise_distance(feats_A, feats_B).numpy())

        return np.squeeze(np.array(distances))


def match_precomputed_points_bipartite(descriptors_A, descriptors_B):
    """
    Matches the Cartesian product of descriptors (bipartite or complete matching, if B is None)
    Some may be None, then their distance is NaN
    """
    with torch.no_grad():

        def assemble(descriptors):
            try:
                nfeat = next(filter(lambda x: x is not None, descriptors)).shape[1]
            except StopIteration:
                return None
            feats = torch.full((len(descriptors), nfeat), np.nan, dtype=torch.float64)
            for i, f in enumerate(descriptors):
                if f is not None:
                    feats[i, :] = f
            return feats

        feats_A = assemble(descriptors_A)

        if descriptors_B is not None:
            feats_B = assemble(descriptors_B)
        else:
            feats_B = feats_A
            descriptors_B = descriptors_A

        if feats_A is None or feats_B is None:
            return np.full((len(descriptors_A), len(descriptors_B)), np.nan)
        else:
            return bag_distances(feats_A, feats_B).numpy()


def bag_euclidean_distances2(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    (https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2)
    """
    x_norm2 = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm2 = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm2 = x_norm2.view(1, -1)

    dist = x_norm2 + y_norm2 - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, min=0)


def bag_distances(x, y):
    if x.shape[0] == 1:
        return nnf.pairwise_distance(x, y)
    else:
        # eps because derivative of sqrt at 0 is nan .. but no gradient if vectors identical due to clamping
        return torch.sqrt(bag_euclidean_distances2(x, y) + 1e-8)
