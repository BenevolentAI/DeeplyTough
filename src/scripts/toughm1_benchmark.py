import argparse
import pickle
import os
import logging
from datasets import ToughM1
from matchers import DeeplyTough, ToughOfficials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Output directory for result pickle')
    parser.add_argument('--alg', type=str, default='DeeplyTough', help='Algorithm type')
    parser.add_argument('--net', type=str, default='', help='DeeplyTough network filepath')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:0')
    parser.add_argument('--nworkers', default=1, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--cvfold', default=0, type=int, help='Fold left-out for testing in leave-one-out setting')
    parser.add_argument('--cvseed', default=7, type=int, help='Dataset shuffling seed')
    parser.add_argument('--num_folds', default=5, type=int, help='Num folds')
    parser.add_argument('--db_split_strategy', default='uniprot_folds', help="pdb_folds|uniprot_folds")
    parser.add_argument('--db_preprocessing', default=0, type=int, help='Bool: whether to run preprocessing for the dataset')

    return parser.parse_args()


def main():
    args = get_cli_args()

    database = ToughM1()
    if args.db_preprocessing:
        database.preprocess_once()
    _, entries = database.get_structures_splits(args.cvfold, strategy=args.db_split_strategy, n_folds=args.num_folds, seed=args.cvseed)

    if args.alg == 'DeeplyTough':
        matcher = DeeplyTough(args.net, device=args.device, batch_size=args.batch_size, nworkers=args.nworkers)
        if matcher.args.seed != args.cvseed or matcher.args.cvfold != args.cvfold:
            logger.warning('Likely not evaluating on the test set for this network')
        entries = matcher.precompute_descriptors(entries)
    elif args.alg == 'OfiGlosa':
        matcher = ToughOfficials('G-LoSA', 2)
    elif args.alg == 'OfiApoc':
        matcher = ToughOfficials('APoc', 2)
    elif args.alg == 'OfiSiteEngine':
        matcher = ToughOfficials('SiteEngine', 3)
    else:
        raise NotImplementedError

    results = database.evaluate_matching(entries, matcher)
    results['benchmark_args'] = args

    fname = 'ToughM1-{}-{}.pickle'.format(args.alg, os.path.basename(args.net).replace('.pth.tar',''))
    os.makedirs(args.output_dir, exist_ok=True)
    pickle.dump(results, open(os.path.join(args.output_dir, fname), 'wb'))
    with open(os.path.join(args.output_dir, fname.replace('.pickle','.csv')), 'w') as f:
        for p, s in zip(results['pairs'], results['scores']):
            f.write('{},{},{}\n'.format(p[0]['code5'], p[1]['code5'], s))
    print('Testing finished, AUC = {}'.format(results['auc']))


if __name__ == '__main__':
    main()
