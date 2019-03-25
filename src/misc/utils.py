import os
import numpy as np
import Bio.PDB as PDB
from scipy.spatial.qhull import QhullError
from scipy.spatial import ConvexHull
import subprocess
import tempfile
import htmd.molecule.molecule as htmdmol
import htmd.molecule.voxeldescriptors as htmdvox

import logging
logger = logging.getLogger(__name__)


def failsafe_hull(coords):
    """
    Wrapper of ConvexHull which returns None if the hull cannot be computed for given points (e.g. all colinear or too few)

    :param coords:
    :return:
    """
    coords = np.array(coords)
    if coords.shape[0] > 3:
        try:
            return ConvexHull(coords)
        except QhullError as e:
            if 'hull precision error' not in str(e) and 'input is less than 3-dimensional' not in str(e):
                raise e
    return None


def hull_centroid_3d(hull):
    """
    The centroid of a 3D polytope. Taken over from http://www.alecjacobson.com/weblog/?p=3854 and
    http://www2.imperial.ac.uk/~rn/centroid.pdf. For >nD ones, https://stackoverflow.com/questions/4824141/how-do-i-calculate-a-3d-centroid

    :param hull: scipy.spatial.ConvexHull
    :return:
    """
    if hull is None:
        return None

    A = hull.points[hull.simplices[:, 0], :]
    B = hull.points[hull.simplices[:, 1], :]
    C = hull.points[hull.simplices[:, 2], :]
    N = np.cross(B-A, C-A)

    # get consistent outer orientation (compensate for the lack of ordering in scipy's facetes), assume a convex hull
    M = np.mean(hull.points[hull.vertices, :], axis=0)
    sign = np.sign(np.sum((A - M) * N, axis=1, keepdims=True))
    N = N * sign

    vol = np.sum(N*A)/6
    centroid = 1/(2*vol)*(1/24 * np.sum(N*((A+B)**2 + (B+C)**2 + (C+A)**2), axis=0))
    return centroid


def point_in_hull(point, hull, tolerance=1e-12):
    # https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)


def structure_to_coord(structure, allow_off_chain=False, allow_hydrogen=False):
    coords = []
    for chains in structure:
        for chain in chains:
             if allow_off_chain or chain.get_id().strip() != '':
                for residue in chain:
                    for atom in residue:
                        if allow_hydrogen or atom.get_name()[0] != 'H':
                            coords.append(atom.get_coord())
    return np.array(coords)


class NonUniqueStructureBuilder(PDB.StructureBuilder.StructureBuilder):
    """This makes PDB more forgiving by being able to load atoms with non-unique names within a residue."""

    @staticmethod
    def _number_to_3char_name(n):
        code = ''
        for k in range(3):
            r = n % 36
            code = chr(ord('A')+r if r<26 else ord('0')+r-26) + code
            n = n // 36
        assert n == 0, 'number cannot fit 3 characters'
        return code

    def init_atom(self, name, coord, b_factor, occupancy, altloc, fullname, serial_number=None, element=None):

        for attempt in range(10000):
            try:
                return super().init_atom(name, coord, b_factor, occupancy, altloc, fullname, serial_number, element)
            except PDB.PDBExceptions.PDBConstructionException:
                name = name[0] + self._number_to_3char_name(attempt)


def center_from_pdb_file(filepath):
    """ Returns the geometric center of a PDB-file structure """
    parser = PDB.PDBParser(PERMISSIVE=1, QUIET=True, structure_builder=NonUniqueStructureBuilder())
    try:
        pocket = parser.get_structure('pocket', filepath)
    except FileNotFoundError:
        return None
    coords = structure_to_coord(pocket, allow_off_chain=True, allow_hydrogen=True)
    if len(coords) > 3:
        return hull_centroid_3d(failsafe_hull(coords))
    elif len(coords) > 0:
        return np.mean(coords, axis=0)
    else:
        return None


def htmd_featurizer(pdb_entries, skip_existing=True):
    """ Ensures than all entries have their HTMD featurization precomputed """
    # note: this is massively hacky but the data also tends to be quite dirty...

    # - Mgltools is Python 2.5 only script destroying Python3 environments, so we have to call it externally in another conda environment
    # - unaddressed warnings info: http://mgldev.scripps.edu/pipermail/mglsupport/2008-December/000091.html
    # - note: http://autodock.scripps.edu/faqs-help/how-to/how-to-prepare-a-receptor-file-for-autodock4
    # - note: http://mgldev.scripps.edu/pipermail/autodock/2008-April/003946.html
    mgl_command = 'source activate deeplytough_mgltools; pythonsh $CONDA_PREFIX/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r {} -U nphs_lps_waters -A hydrogens'

    for entry in pdb_entries:
        pdb_path = entry['protein']
        npz_path = entry['protein_htmd']
        if skip_existing and os.path.exists(npz_path):
            continue

        output_dir = os.path.dirname(npz_path)
        os.makedirs(output_dir, exist_ok=True)
        logging.info('Pre-processing {} with HTMD...'.format(pdb_path))

        def compute_channels():
            pdbqt_path = os.path.join(output_dir, os.path.basename(pdb_path)) + 'qt'
            if not os.path.exists(pdbqt_path) and os.path.exists(pdbqt_path.replace('.pdb', '_model1.pdb')):
                os.rename(pdbqt_path.replace('.pdb', '_model1.pdb'), pdbqt_path)
            mol = htmdmol.Molecule(pdbqt_path)
            mol.filter('protein')  # take only on-chain atoms

            # slaughtered getVoxelDescriptors()
            channels = htmdvox._getAtomtypePropertiesPDBQT(mol)
            sigmas = htmdvox._getRadii(mol)
            channels = sigmas[:, np.newaxis] * channels.astype(float)
            coords = mol.coords[:, :, mol.frame]

            np.savez(npz_path, channels=channels, coords=coords)

        try:
            subprocess.run(['/bin/bash', '-c', mgl_command.format(pdb_path)], cwd=output_dir, check=True)
            compute_channels()
        except Exception as err1:
            try:
                # Put input through obabel to handle some problematic formattings, it's parser seems quite robust
                # (could actually directly go to pdbqt here with `-xr -xc -h` but somehow the partial charges are all zero)
                with tempfile.TemporaryDirectory() as tmpdirname:
                    pdb2_path = os.path.join(tmpdirname, os.path.basename(pdb_path))
                    subprocess.run(['obabel', pdb_path, '-O', pdb2_path, '-h'], check=True)
                    subprocess.run(['/bin/bash', '-c', mgl_command.format(pdb2_path)], cwd=output_dir, check=True)
                compute_channels()
            except Exception as err2:
                logger.exception(err2)
