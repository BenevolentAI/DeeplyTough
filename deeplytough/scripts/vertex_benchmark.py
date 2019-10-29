import argparse
import pickle
import os
import logging
from datasets import Vertex
from matchers import DeeplyTough

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
    parser.add_argument('--db_preprocessing', default=0, type=int, help='Bool: whether to run preprocessing for the dataset')
    
    return parser.parse_args()


def main():
    args = get_cli_args()

    database = Vertex()
    if args.db_preprocessing:
        database.preprocess_once()
    entries = database.get_structures()

    if args.alg == 'DeeplyTough':
        matcher = DeeplyTough(args.net, device=args.device, batch_size=args.batch_size, nworkers=args.nworkers)
        entries = matcher.precompute_descriptors(entries)
    else:
        raise NotImplementedError

    results = database.evaluate_matching(entries, matcher)
    results['benchmark_args'] = args
    results['entries'] = entries  # includes descriptors

    fname = 'Vertex-{}-{}.pickle'.format(args.alg, os.path.basename(args.net).replace('.pth.tar',''))
    os.makedirs(args.output_dir, exist_ok=True)
    pickle.dump(results, open(os.path.join(args.output_dir, fname), 'wb'))
    with open(os.path.join(args.output_dir, fname.replace('.pickle','.csv')), 'w') as f:
        for p, s in zip(results['pairs'], results['scores']):
            f.write('{},{},{}\n'.format(p[0], p[1], s))
    print('Testing finished, AUC = {}'.format(results['auc']))

if __name__ == '__main__':
    main()
