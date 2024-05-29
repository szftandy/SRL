import os
import sys
import argparse
import logging
import random
import time
import torch
from pprint import pprint

def get_configs():
    parser = argparse.ArgumentParser()
    # General Configs
    # parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--train", action="store_true", default=False)
    # parser.add_argument("--test", action="store_true", default=True)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--pred", action="store_true", default=True)
    parser.add_argument("--stable", action="store_true", default=False)
    # parser.add_argument("--data", default="yago")
    # parser.add_argument("--data", default="fb15k-237")
    # parser.add_argument("--data", default="family")
    parser.add_argument("--data", default="dblp")
    parser.add_argument("--sample_pairs", type=int, default=10000)
    parser.add_argument("--gpu", type=int, default=0)
    # parser.add_argument("--stable_gpu", type=int, default=-1)
    parser.add_argument("--stable_gpu", type=int, default=2)
    # Learning Configs
    parser.add_argument("--learn_path_len", type=int, default=3)
    parser.add_argument("--test_path_len", type=int, default=3)
    parser.add_argument("--emb_size", type=int, default=1024)
    parser.add_argument("--sparsity", type=float, default=1)
    parser.add_argument("--seed", type=int, default=2077)
    parser.add_argument("--sample_worker", type=int, default=10)
    # Training Configs
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=3000)
    parser.add_argument("--stable_model", type=str, default='lin', help='lin/nonlin')
    # Stable DWR Configs
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--decorr_steps", type=int, default=100)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--lr_dwr', type=float, default=0.01)
    # StableNet Configs
    parser.add_argument("--epoch_stable", type=int, default=20)
    parser.add_argument("--epoch_pre", type=int, default=0)
    parser.add_argument('--n_feature', type=int, default=128, help = 'number of pre-saved features')
    parser.add_argument('--lr_stable', type=float, default=1.0, help = 'learning rate of balance')
    parser.add_argument('--lambda_decay_rate', type=float, default=1, help = 'ratio of epoch for lambda to decay')
    parser.add_argument('--lambda_decay_epoch', type=int, default=5, help = 'number of epoch for lambda to decay')
    parser.add_argument('--lambda_p', type = float, default=70.0, help = 'weight decay for weight1 ')
    parser.add_argument('--decay_pow', type=float, default=2, help = 'value of pow for weight decay')
    parser.add_argument('--min_lambda_times', type=float, default=0.01, help = 'number of global table levels')
    # Prediction Configs
    parser.add_argument("--sort_relwise", action="store_true", default=True)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0)
    args = parser.parse_args()
    args.stable_gpu = args.gpu if args.stable_gpu == -1 else args.stable_gpu
    return args

def get_logger(args):
    logger = logging.getLogger()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt); logger.addHandler(console)
    logfile = logging.FileHandler(args.log_dir, 'w')
    logfile.setFormatter(fmt); logger.addHandler(logfile)
    return logger

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def set_configure(use_logger=True):
    args = get_configs()
    if args.train:
        # get and create output dir
        strTime = time.strftime("%m-%d-%H-%M-%S", time.localtime(int(time.time())))
        args.output_dir = f'output/{args.data}/{strTime}/'
        os.makedirs(args.output_dir)
        # find log dir and rule dir
        args.log_dir = args.output_dir+'log.txt'
        args.rule_dir = f'rule/{args.data}/{strTime}/'
        # output config
        options_print = {u:v for u,v in vars(args).items() if type(v) is not dict}
        pprint(options_print, stream=open(args.output_dir+'config.txt', 'w'))
    if not args.train and args.test: # Load last model in dirs
        # find output dir
        base_outdir = f'output/{args.data}/'
        model_dirs = sorted([i for i in os.listdir(base_outdir) if os.path.isfile(f'{base_outdir}/{i}/model.pkt')])
        args.output_dir = f'output/{args.data}/{model_dirs[-1]}/'
        # find log dir
        args.log_dir = args.output_dir+'test_log.txt'
        # get rule dir
        strTime = time.strftime("%m-%d-%H-%M-%S", time.localtime(int(time.time())))
        args.rule_dir = f'rule/{args.data}/{strTime}/'
    if not args.train and not args.test: # Load last rule
        # find log dir and rule dir
        base_ruledir = f'rule/{args.data}/'
        rule_dirs = sorted([i for i in os.listdir(base_ruledir) if os.listdir(f'{base_ruledir}/{i}')])
        args.rule_dir = base_ruledir + f'{rule_dirs[-1]}/'
        args.log_dir = args.rule_dir+'pred_log.txt'
    if use_logger:
        logger = get_logger(args)
    # set_seed(args.seed)
    return args, logger