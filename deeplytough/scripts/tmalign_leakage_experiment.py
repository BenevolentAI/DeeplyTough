import argparse
import logging
import os
import pickle
import concurrent.futures
from functools import partial

from datasets import ToughM1, Vertex, Prospeccts
from matchers.tmalign import make_full_pocket, make_tough_lig_pocket, align_worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='.', help='Output directory for result pickle')
    parser.add_argument('--cvfold', default=0, type=int, help='Fold left-out for testing in leave-one-out setting')
    parser.add_argument('--num_folds', default=5, type=int, help='Num folds')
    parser.add_argument('--db_split_strategy', default='seqclust', help="pdb_folds|uniprot_folds|seqclust|none")
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')
    parser.add_argument('--tough_lig_pocket', default=0, type=int, help='Bool: whether to use induced (True) or detected pockets (False).')

    args = parser.parse_args()
    return args


def main():
    args = get_cli_args()
    
    tough_fixer = make_tough_lig_pocket if args.tough_lig_pocket else make_full_pocket

    # Tough
    for seed in range(1, 11):
        train_entries, test_entries = create_tough_splits(
            db_split_strategy=args.db_split_strategy, fold_nr=args.cvfold, n_folds=args.num_folds, seed=seed,
            exclude_Vertex_from_train=False, exclude_Prospeccts_from_train=False
        )
        for e in train_entries:
            e['pocket'] = tough_fixer(e)
        for e in test_entries:
            e['pocket'] = tough_fixer(e)
        
        scores = align_entries(train_entries, test_entries)
        pickle.dump(scores, open(os.path.join(args.output_dir, f'tmalign_tough_seed{seed}_{args.db_split_strategy}_{args.tough_lig_pocket}.pickle'), 'wb'))

    # Prospeccts
    train_entries, _ = create_tough_splits(
        db_split_strategy='none', fold_nr=args.cvfold, n_folds=args.num_folds, seed=args.seed,
        exclude_Vertex_from_train=False, exclude_Prospeccts_from_train=args.db_split_strategy
    )
    for e in train_entries:
        e['pocket'] = tough_fixer(e)
    
    for dbname in Prospeccts.dbnames:
        database = Prospeccts(dbname)
        pro_entries = database.get_structures()

        scores = align_entries(train_entries, pro_entries)
        pickle.dump(scores, open(os.path.join(args.output_dir, f'tmalign_prospeccts{dbname}_{args.db_split_strategy}_{args.tough_lig_pocket}.pickle'), 'wb'))

    # Vertex
    train_entries, _ = create_tough_splits(
        db_split_strategy='none', fold_nr=args.cvfold, n_folds=args.num_folds, seed=args.seed,
        exclude_Vertex_from_train=args.db_split_strategy, exclude_Prospeccts_from_train=False
    )
    for e in train_entries:
        e['pocket'] = tough_fixer(e)
    
    database = Vertex()
    vertex_entries = database.get_structures()

    scores = align_entries(train_entries, vertex_entries)
    pickle.dump(scores, open(os.path.join(args.output_dir, f'tmalign_vertex_{args.db_split_strategy}_{args.tough_lig_pocket}.pickle'), 'wb'))    


def create_tough_splits(db_split_strategy, fold_nr, n_folds=5, seed=0, exclude_Vertex_from_train=False,
                         exclude_Prospeccts_from_train=False):

    pdb_train, pdb_test = ToughM1().get_structures_splits(fold_nr, strategy=db_split_strategy,
                                                          n_folds=n_folds, seed=seed)

    # Vertex
    if exclude_Vertex_from_train:
        # Get Vertex dataset
        vertex = Vertex().get_structures()

        # Exclude entries from tough training set that exist in the vertex set
        logger.info(f'Before Vertex filter {len(pdb_train)}')
        if exclude_Vertex_from_train == 'uniprot':
            vertex_ups = set([entry['uniprot'] for entry in vertex] + ['None'])
            pdb_train = list(filter(lambda entry: entry['uniprot'] not in vertex_ups, pdb_train))
        elif exclude_Vertex_from_train == 'pdb':
            vertex_pdbs = set([entry['code'] for entry in vertex])
            pdb_train = list(filter(lambda entry: entry['code'] not in vertex_pdbs, pdb_train))
        elif exclude_Vertex_from_train == 'seqclust':
            vertex_seqclusts = set([c for entry in vertex for c in entry['seqclusts']] + ['None'])
            pdb_train = list(filter(lambda entry: entry['seqclust'] not in vertex_seqclusts, pdb_train))
        else:
            raise NotImplementedError()
        logger.info(f'After Vertex filter {len(pdb_train)}')

    # ProSPECCTS
    if exclude_Prospeccts_from_train:
        # Get ProSPECCTs datasets
        all_prospeccts = [entry for dbname in Prospeccts.dbnames for entry in Prospeccts(dbname).get_structures()]

        # Exclude entries from tough training set that exist in the ProSPECCTs sets
        logger.info(f'Before Prospeccts filter {len(pdb_train)}')
        if exclude_Prospeccts_from_train == 'uniprot':
            prospeccts_ups = set([u for entry in all_prospeccts for u in entry['uniprot']] + ['None'])
            pdb_train = list(filter(lambda entry: entry['uniprot'] not in prospeccts_ups, pdb_train))
        elif exclude_Prospeccts_from_train == 'pdb':
            prospeccts_pdbs = set([entry['code'] for entry in all_prospeccts])
            pdb_train = list(filter(lambda entry: entry['code'].lower() not in prospeccts_pdbs, pdb_train))
        elif exclude_Prospeccts_from_train == 'seqclust':
            prospeccts_seqclusts = set([c for entry in all_prospeccts for c in entry['seqclusts']] + ['None'])
            pdb_train = list(filter(lambda entry: entry['seqclust'] not in prospeccts_seqclusts, pdb_train))
        else:
            raise NotImplementedError()
        logger.info(f'After Prospeccts filter {len(pdb_train)}')

    return pdb_train, pdb_test


def align_entries(trainset, testset):
    results = []    
    for te in testset:
        if not os.path.exists(te['pocket']):
            print(f"Test pocket {te['pocket']} not found, skipping.")
            continue
        best_score = -1
        best_entry = None
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for score, tr in executor.map(partial(align_worker, te=te), trainset):
                if score > best_score:
                    best_score = score
                    best_entry = tr
        if best_entry is not None:
            print(best_score, best_entry['pocket'], te['pocket'])
            results.append([best_score, best_entry['pocket'], te['pocket']])
    return results


if __name__ == '__main__':
    main()
