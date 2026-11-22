"""Run several independent SNN re-training/testing runs for one perturbation configuration.

This is a batch driver around AgentSiameseNetwork (paper step 3). The data loaders are
built once and shared across all runs, which avoids re-decoding ~34k PNGs for every seed.
Each run still re-seeds the RNGs and re-initializes the network, the optimizer and the
early-stopping state, so the runs remain independent as required by the paper.
"""

import os
import copy
import json
import time
import argparse

import torch
import torch.optim as optim

from utils import utils
from utils.EarlyStopping import EarlyStopping
from networks.SiameseNetwork import SiameseNetwork
from networks.SiameseDenseNet import SiameseDenseNet
from agents.AgentSiameseNetwork import AgentSiameseNetwork


# Section 2.2 initializes the verifier inside PriCheXy-Net from a 99.4% verification model,
# while section 3.1 says that the "incorporated SNN" is retrained without restating its
# initialization. The released evaluation code starts from ImageNet. We retain both defensible
# readings instead of claiming that the paper unambiguously specifies either one.
SNN_INIT_CHOICES = ('imagenet', 'pretrained_ver')
PRETRAINED_VER_MODEL = './networks/pretrained_verification_model.pth'


def reset_for_run(agent, description, seed, snn_init='imagenet', snn_arch='resnet50'):
    """Put the agent back into a pristine state so the next run is independent."""

    utils.seed_all(seed)

    agent.config['experiment_description'] = description
    agent.SAVINGS_PATH = './archive/' + description + '/'
    os.makedirs(agent.SAVINGS_PATH, exist_ok=True)

    # snn_arch selects the attacker construction. 'densenet121_cxr' is the second attacker of
    # PROTOCOL_ADDENDUM section C: a different backbone AND a different pretraining domain, so
    # that the audit's conclusions can be tested for architecture dependence.
    if snn_arch == 'densenet121_cxr':
        agent.net = SiameseDenseNet('./networks/pretrained_classifier.pth').cuda()
    else:
        agent.net = SiameseNetwork().cuda()
    if snn_init == 'pretrained_ver':
        if snn_arch != 'resnet50':
            raise SystemExit(
                "--snn_init pretrained_ver loads the ResNet-50 verification checkpoint and is "
                "incompatible with --snn_arch " + snn_arch + ". The DenseNet attacker is "
                "already initialized from a chest-radiograph model by construction, which is "
                "its equivalent of the adapted arm.")
        agent.net.load_state_dict(torch.load(PRETRAINED_VER_MODEL))
    agent.best_net = copy.deepcopy(agent.net)
    agent.optimizer = optim.Adam(agent.net.parameters(), lr=agent.learning_rate)

    agent.start_epoch = 0
    agent.es = EarlyStopping(patience=agent.early_stopping)
    agent.best_loss = 100000
    agent.loss_dict = {'training': [], 'validation': []}


def main():
    parser = argparse.ArgumentParser('Batch patient verification runs')
    parser.add_argument('--config_path', default='./config_files/')
    parser.add_argument('--config', default='config_retrainSNN.json')
    parser.add_argument('--tag', required=True, help='Short name of the configuration, e.g. ff_mu0.01')
    parser.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--rebuild_data_per_seed', action='store_true',
                        help='Rebuild the datasets for every seed. Required for DP-Pix, whose '
                             'noise is drawn while the dataset is being built.')
    parser.add_argument('--snn_arch', default='resnet50',
                        choices=['resnet50', 'densenet121_cxr'],
                        help='attacker backbone; densenet121_cxr is the second construction')
    parser.add_argument('--snn_init', default='imagenet', choices=SNN_INIT_CHOICES,
                        help="How to initialize the attacker. 'imagenet' matches the released "
                             "code; 'pretrained_ver' tests the alternative reading that retains "
                             "the incorporated 99.4 %% verification model's initialization.")
    args = parser.parse_args()

    with open(args.config_path + args.config, 'r') as f:
        config = json.loads(f.read())

    if 'BATCH_SIZE' in os.environ:
        config['batch_size'] = int(os.environ['BATCH_SIZE'])
    if 'LEARNING_RATE' in os.environ:
        config['learning_rate'] = float(os.environ['LEARNING_RATE'])

    seeds = [int(s) for s in args.seeds.split(',')]

    # Skip seeds whose results file already exists, so the batch can be resumed after an abort
    pending = [s for s in seeds
               if not os.path.isfile(f'./archive/ver_{args.tag}_seed{s}/ver_{args.tag}_seed{s}_results.txt')]
    print(f'[{args.tag}] snn_init:        {args.snn_init}', flush=True)
    print(f'[{args.tag}] requested seeds: {seeds}', flush=True)
    print(f'[{args.tag}] pending seeds:   {pending}', flush=True)
    if not pending:
        print(f'[{args.tag}] nothing to do', flush=True)
        return

    agent = None
    for seed in pending:
        description = f'ver_{args.tag}_seed{seed}'
        started = time.time()
        print(f'\n===== [{args.tag}] seed {seed} starting at {time.strftime("%F %T")} =====', flush=True)

        # AgentSiameseNetwork:33 hard-codes utils.seed_all(42) and runs it BEFORE the data
        # loaders are built. For DP-Pix the Laplace noise is drawn during that build, so
        # without this override every seed would get seed 42's noise and
        # --rebuild_data_per_seed would sample nothing. seed_all() reads PRICHEXY_SEED.
        os.environ['PRICHEXY_SEED'] = str(seed)

        if agent is None or args.rebuild_data_per_seed:
            # Building the agent also builds the three data loaders (the expensive part)
            utils.seed_all(seed)
            config['experiment_description'] = description
            os.makedirs('./archive/' + description, exist_ok=True)
            agent = AgentSiameseNetwork(config)

        reset_for_run(agent, description, seed, args.snn_init, args.snn_arch)
        torch.cuda.reset_peak_memory_stats()
        agent.run()

        # The box is shared, so record what this run actually needed rather than guessing.
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
        print(f'[{args.tag}] seed {seed} peak GPU: {peak:.2f} GiB allocated, '
              f'{reserved:.2f} GiB reserved', flush=True)

        minutes = (time.time() - started) / 60
        print(f'===== [{args.tag}] seed {seed} done in {minutes:.1f} min =====', flush=True)
        with open('./archive/snn_runs/verification_progress.log', 'a') as log:
            log.write(f'{time.strftime("%F %T")}\t{args.tag}\tseed={seed}\t{minutes:.1f} min\n')


if __name__ == '__main__':
    main()
