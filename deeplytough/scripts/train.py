import argparse
import ast
import datetime
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as nnf
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from engine.datasets import create_tough_dataset
from engine.models import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cli_args():
    parser = argparse.ArgumentParser()

    # Optimization arguments
    parser.add_argument('--wd', default=5.0e-4, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.2, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default='[100,150]', help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--epochs', default=151, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--batch_parts', default=1, type=int, help='Batch can be evaluated sequentially in multiple shards, >=1, very useful in low memory settings, though computation is not strictly equivalent due to batch normalization runnning statistics.')
    parser.add_argument('--optim', default='adam', help='Optimizer: sgd|adam')
    parser.add_argument('--max_train_samples', default=2500, type=int, help='Max train samples per epoch (good for large datasets)')
    parser.add_argument('--max_test_samples', default=100, type=int, help='Max test samples per epoch (good for large datasets)')

    # Experiment arguments
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--nworkers', default=3, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--resume', default='', help='Loads a previously saved model.')

    # Dataset
    parser.add_argument('--output_dir', default='results', help='Directory to store results')
    parser.add_argument('--cvfold', default=0, type=int, help='Fold left-out for testing in leave-one-out setting')
    parser.add_argument('--num_folds', default=5, type=int, help='Num folds')
    parser.add_argument('--augm_rot', default=1, type=int, help='Training augmentation: Bool, random rotation')
    parser.add_argument('--augm_mirror_prob', default=0, type=float, help='Training augmentation: Probability of mirroring about axes')
    parser.add_argument('--augm_sampling_dist', default=2.0, type=float, help='Training augmentation: Max distance from fpocket centers')
    parser.add_argument('--augm_decoy_prob', default=0.1, type=float, help='Training augmentation: Probability of negative decoy')
    parser.add_argument('--patch_size', default=24, type=int, help='Patch size for training')
    parser.add_argument('--input_normalization', default=1, type=int, help='Bool: whether to normalize input statistics')
    parser.add_argument('--db_exclude_vertex', default='', type=str, help='Whether to exclude Vertex dataset proteins in the training fold: (|seqclust|uniprot|pdb)')
    parser.add_argument('--db_exclude_prospeccts', default='', type=str, help='Whether to exclude Prospeccts dataset proteins in the training fold: (|seqclust|uniprot|pdb)')
    parser.add_argument('--db_split_strategy', default='seqclust', help="pdb_folds|uniprot_folds|seqclust|none")
    parser.add_argument('--db_preprocessing', default=0, type=int, help='Bool: whether to run preprocessing for the dataset')
    parser.add_argument('--db_size_limit', default=0, type=int, help='Artification restriction of database size, either on # pdbs (>0) or # pairs (<0)')

    # Model
    parser.add_argument('--model_config', default='se_16_16_16_16_7_3_2_batch_1,se_32_32_32_32_3_1_1_batch_1,se_48_48_48_48_3_1_2_batch_1,se_64_64_64_64_3_0_1_batch_1,se_256_0_0_0_3_0_2_batch_1,r,b,c_128_1', help='Defines the model as a sequence of layers.')
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')
    parser.add_argument('--l2_normed_descriptors', default=1, type=int, help='L2-normalize descriptors/network outputs')
    parser.add_argument('--loss_margin', default=1.0, type=float, help='Margin in hinge losses')
    parser.add_argument('--stability_loss_weight', default=1.0, type=float, help='Weight of augmentation invariance loss')
    parser.add_argument('--stability_loss_squared', default=0, type=int, help='Augmentation invariance loss distances squared (1) or not (0)')

    args = parser.parse_args()
    args.start_epoch = 0
    args.lr_steps = ast.literal_eval(args.lr_steps)
    args.output_dir = args.output_dir.replace('TTTT', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    assert args.batch_size % args.batch_parts == 0
    return args


def main():
    args = get_cli_args()
    print('Will save to ' + args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'cmdline.txt'), 'w') as f:
        f.write(" ".join(["'"+a+"'" if (len(a) == 0 or a[0] != '-') else a for a in sys.argv]))

    set_seed(args.seed)
    device = torch.device(args.device)
    writer = SummaryWriter(args.output_dir)

    train_dataset, test_dataset = create_tough_dataset(
        args, fold_nr=args.cvfold, n_folds=args.num_folds, seed=args.seed,
        exclude_Vertex_from_train=args.db_exclude_vertex, exclude_Prospeccts_from_train=args.db_exclude_prospeccts
    )
    logger.info('Train set size: %d, test set size: %d', len(train_dataset), len(test_dataset))

    # Create model and optimizer (or resume pre-existing)
    if args.resume != '':
        if args.resume == 'RESUME':
            args.resume = args.output_dir + '/model.pth.tar'
        model, optimizer, scheduler = resume(args, train_dataset, device)
    else:
        model = create_model(args, train_dataset, device)
        if args.input_normalization:
            model.set_input_scaler(estimate_scaler(args, train_dataset, nsamples=200))
        optimizer = create_optimizer(args, model)
        scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_decay)

    ############
    def train():
        model.train()

        loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size//args.batch_parts,
                                             num_workers=args.nworkers, shuffle=True, drop_last=True,
                                             worker_init_fn=set_worker_seed)

        if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
            loader = tqdm(loader, ncols=100)

        loss_buffer, loss_stabil_buffer, pos_dist_buffer, neg_dist_buffer = [], [], [], []
        t0 = time.time()

        for bidx, batch in enumerate(loader):
            if 0 < args.max_train_samples < bidx * args.batch_size//args.batch_parts:
                break
            t_loader = 1000*(time.time()-t0)

            inputs = batch['inputs'].to(device)  # dimensions: batch_size x (4 or 2) x 24 x 24 x 24
            targets = batch['targets'].to(device)

            if bidx % args.batch_parts == 0:
                optimizer.zero_grad()
            t0 = time.time()

            outputs = model(inputs.view(-1, *inputs.shape[2:]))
            outputs = outputs.view(*inputs.shape[:2], -1)
            loss_joint, loss_match, loss_stabil, pos_dist, neg_dist = compute_loss(args, outputs, targets, True)
            loss_joint.backward()

            if bidx % args.batch_parts == args.batch_parts-1:
                if args.batch_parts > 1:
                    for p in model.parameters():
                        p.grad.data.div_(args.batch_parts)
                optimizer.step()

            t_trainer = 1000*(time.time()-t0)
            loss_buffer.append(loss_match.item())
            loss_stabil_buffer.append(loss_stabil.item() if isinstance(loss_stabil, torch.Tensor) else loss_stabil)
            pos_dist_buffer.extend(pos_dist.cpu().numpy().tolist())
            neg_dist_buffer.extend(neg_dist.cpu().numpy().tolist())
            logger.debug('Batch loss %f, Loader time %f ms, Trainer time %f ms.', loss_buffer[-1], t_loader, t_trainer)
            t0 = time.time()

        ret = {'loss': np.mean(loss_buffer), 'loss_stabil': np.mean(loss_stabil_buffer),
               'pos_dist': np.mean(pos_dist_buffer), 'neg_dist': np.mean(neg_dist_buffer)}
        return ret

    ############
    def test():
        model.eval()

        loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size//args.batch_parts,
                                             num_workers=args.nworkers, worker_init_fn=set_worker_seed)

        if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
            loader = tqdm(loader, ncols=100)

        loss_buffer, loss_stabil_buffer, pos_dist_buffer, neg_dist_buffer = [], [], [], []

        with torch.no_grad():
            for bidx, batch in enumerate(loader):
                if 0 < args.max_test_samples < bidx * args.batch_size//args.batch_parts:
                    break
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)

                outputs = model(inputs.view(-1, *inputs.shape[2:]))
                outputs = outputs.view(*inputs.shape[:2], -1)
                loss_joint, loss_match, loss_stabil, pos_dist, neg_dist = compute_loss(args, outputs, targets, False)

                loss_buffer.append(loss_match.item())
                loss_stabil_buffer.append(loss_stabil.item() if isinstance(loss_stabil, torch.Tensor) else loss_stabil)
                pos_dist_buffer.extend(pos_dist.cpu().numpy().tolist())
                neg_dist_buffer.extend(neg_dist.cpu().numpy().tolist())

        return {'loss': np.mean(loss_buffer), 'loss_stabil': np.mean(loss_stabil_buffer),
                'pos_dist': np.mean(pos_dist_buffer), 'neg_dist': np.mean(neg_dist_buffer)}

    ############
    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        print(f'Epoch {epoch}/{args.epochs} ({args.output_dir}):')
        scheduler.step()

        train_stats = train()
        for k, v in train_stats.items():
            writer.add_scalar('train/' + k, v, epoch)
        print(f"-> Train distances: p {train_stats['pos_dist']}, n {train_stats['neg_dist']}, \tLoss: {train_stats['loss']}")

        if (epoch+1) % args.test_nth_epoch == 0 or epoch+1 == args.epochs:
            test_stats = test()
            for k, v in test_stats.items():
                writer.add_scalar('test/' + k, v, epoch)
            print(f"-> Test distances: p {test_stats['pos_dist']}, n {test_stats['neg_dist']}, \tLoss: {test_stats['loss']}")

        torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
                   os.path.join(args.output_dir, 'model.pth.tar'))

        if math.isnan(train_stats['loss']):
            break


def compute_loss(args, outputs, targets, training):
    """
    Computes both stability and contrastive loss
    """
    outputs = torch.squeeze(outputs)
    targets = torch.squeeze(targets)
    assert outputs.dim() == 3 and targets.dim() == 1

    if args.l2_normed_descriptors:
        outputs = nnf.normalize(outputs, p=2, dim=2)

    # Stability loss
    if training and args.stability_loss_weight > 0:
        # every odd entry in the batch is a perturbed version of the previous even entry
        a = outputs[:, 0::2].view(-1, outputs.shape[-1])
        b = outputs[:, 1::2].view(-1, outputs.shape[-1])
        if args.stability_loss_squared:
            loss_stabil = nnf.pairwise_distance(a, b).pow(2).mean()
        else:
            loss_stabil = nnf.pairwise_distance(a, b).mean()
        # continue with just the even ones
        outputs = outputs[:, 0::2]
    else:
        loss_stabil = 0

    # Contrastive loss
    assert outputs.shape[1] == 2
    dists = nnf.pairwise_distance(outputs[:, 0], outputs[:, 1]).view(-1)

    pos_loss = dists.pow(2)
    neg_loss = torch.clamp(args.loss_margin - dists, min=0).pow(2)
    loss_match = torch.sum(pos_loss * targets + neg_loss * (1 - targets)) / targets.numel()

    loss_joint = loss_match + args.stability_loss_weight * loss_stabil
    return loss_joint, loss_match, loss_stabil, dists[targets > 0.5].detach(), dists[targets < 0.5].detach()


def resume(args, dataset, device):
    """
    Loads model and optimizer state from a previous checkpoint
    """
    print(f"=> loading checkpoint '{args.resume}'")
    checkpoint = torch.load(args.resume, map_location=str(device))

    model = create_model(checkpoint['args'], dataset, device)
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = create_optimizer(args, model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    args.start_epoch = checkpoint['epoch']

    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_decay)
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler


def create_optimizer(args, model):
    if args.optim == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_worker_seed(worker_id):
    np.random.seed(torch.initial_seed() % (2**32 - 1))


def estimate_scaler(args, train_dataset, nsamples):
    logger.info('Estimating dataset normalization')
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    bidx = 0
    with tqdm(total=nsamples) as pbar:
        while True:
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.nworkers,
                                                 shuffle=True, drop_last=True, worker_init_fn=set_worker_seed)
            for batch in loader:
                inputs = batch['inputs'].view(-1, *batch['inputs'].shape[2:])
                assert inputs.dim() == 5
                voxels = inputs.transpose(1, 4).contiguous().view(-1, inputs.shape[1]).numpy()
                scaler.partial_fit(voxels)

                bidx += inputs.shape[0]
                pbar.update(inputs.shape[0])
                if bidx >= nsamples:
                    return scaler


if __name__ == "__main__":
    main()
