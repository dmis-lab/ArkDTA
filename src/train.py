import random
import argparse
import tarfile
import wandb
import pickle
import setproctitle
import os 
import os.path
import json
import numpy as np

from os import kill
from os import getpid
from signal import SIGKILL
import time

from trainer import *

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler

CHOICES_RESIDUE_ENCODER     = [None]
CHOICES_RESIDUE_ADDON       = ['ARKMAB']
CHOICES_LIGELEM_ENCODER     = [None]
CHOICES_LIGELEM_POOLER      = [None]
CHOICES_LIGELEM_POOLTYPE    = [None]
CHOICES_COMPLEX_DECODER     = ['PMA.Residue']
CHOICES_AFFINITY_PREDICTOR  = [None]

CHOICES_ATTENTION_OPTION    = ['additive']
CHOICES_ATTENTION_REGUXN    = [None]

num_cpus = os.cpu_count()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

ngpus_per_node = torch.cuda.device_count()

parser = argparse.ArgumentParser()
# Related to WANDB and Local Path
parser.add_argument('--root_path',       '-rp', default='../',            type=str)
parser.add_argument('--checkpoint_path', '-cp', default='./saved/',       type=str)
parser.add_argument('--project_name',    '-pn', default='defaultproject', type=str)
parser.add_argument('--session_name',    '-sn', default='defaultsession', type=str)
parser.add_argument('--group_name',      '-gn', default='defaultgroup',   type=str)
parser.add_argument('--finetune_name',   '-ft', default='None',           type=str)

# Related to Dataset
parser.add_argument('--dataset_version',   default='221027',           type=str)
parser.add_argument('--dataset_subsets',   default='pdb_2020_refined', type=str) 
parser.add_argument('--dataset_partition', default='random',           type=str, choices=['random', 'randomsingle']) 
parser.add_argument('--dataset_loading',   default='preload',          type=str, choices=['preload', 'autoload'])
parser.add_argument('--dataset_subsample', default=None,               type=int)
parser.add_argument('--ba_measure',        default='KIKD',             type=str)

# Related to Model Execution Mode
parser.add_argument('--pred_model',        default='arkdta',  type=str)
parser.add_argument('--fold_num',          default=0,         type=int)
parser.add_argument('--testing_mode',      default=False,     action='store_true')
parser.add_argument('--analysis_mode',     default=False,     action='store_true')

# Related to Debugging and ToyTesting
parser.add_argument('--debug_mode',  default=False, action='store_true')
parser.add_argument('--debug_index', default=500,   type=int)
parser.add_argument('--toy_test',    default=False, action='store_true')

# Related to DDP and Others
parser.add_argument('--num_cpus',    default=num_cpus,       type=int)
parser.add_argument('--num_gpus',    default=ngpus_per_node, type=int)
parser.add_argument('--port',        default=12345,          type=int)
parser.add_argument('--random_seed', default=24,             type=int)

# Related to Model Hyperparameters
parser.add_argument('--hp_default_setting',  default=False, action='store_true')
parser.add_argument('--hp_num_epochs',       default=100,   type=int)
parser.add_argument('--hp_learning_rate',    default=1e-4,  type=float)
parser.add_argument('--hp_weight_decay',     default=0.0,   type=float)
parser.add_argument('--hp_batch_size',       default=32,    type=int)
parser.add_argument('--hp_early_patience',   default=20,    type=int)
parser.add_argument('--hp_main_coefficient', default=1.0,   type=float)
parser.add_argument('--hp_aux_coefficient',  default=5.0,   type=float)
parser.add_argument('--hp_dropout_rate',     default=0.1,   type=float)
parser.add_argument('--hp_hidden_nodes',     default=128,   type=int)
parser.add_argument('--hp_temp_scalar',      default=10.0,  type=float)

# Related to ArkDTA & PrototypeArkDTA
parser.add_argument('--arkdta_residue_encoder',    default='MLP',         type=str, choices=CHOICES_RESIDUE_ENCODER)
parser.add_argument('--arkdta_residue_addon',      default='ARKMAB',      type=str, choices=CHOICES_RESIDUE_ADDON)
parser.add_argument('--arkdta_ligelem_encoder',    default='MLP',         type=str, choices=CHOICES_LIGELEM_ENCODER)
parser.add_argument('--arkdta_ligelem_pooler',     default='Max',         type=str, choices=CHOICES_LIGELEM_POOLER)
parser.add_argument('--arkdta_ligelem_pooltype',   default='early',       type=str, choices=CHOICES_LIGELEM_POOLTYPE)
parser.add_argument('--arkdta_complex_decoder',    default='PMA.Residue', type=str, choices=CHOICES_COMPLEX_DECODER)
parser.add_argument('--arkdta_affinity_predictor', default='MLP',         type=str, choices=CHOICES_AFFINITY_PREDICTOR)

parser.add_argument('--arkdta_hidden_dim',         default=128,                type=int)
parser.add_argument('--arkdta_cnn_depth',          default=2,                  type=int)
parser.add_argument('--arkdta_kernel_size',        default=5,                  type=int)
parser.add_argument('--arkdta_ecfpvec_dim',        default=1024,               type=int)
parser.add_argument('--arkdta_esm_model',          default='esm2_t6_8M_UR50D', type=str)
parser.add_argument('--arkdta_esm_freeze',         default='False',            type=str)
parser.add_argument('--arkdta_gnn_depth',          default=3,                  type=int)
parser.add_argument('--arkdta_sab_depth',          default=0,                  type=int)
parser.add_argument('--arkdta_num_heads',          default=4,                  type=int)
parser.add_argument('--arkdta_num_seeds',          default=2,                  type=int)
parser.add_argument('--arkdta_attention_option',   default='additive',         type=str, choices=CHOICES_ATTENTION_OPTION)

parser.add_argument('--arkdta_regularization',     default='None',     type=str,   choices=CHOICES_ATTENTION_REGUXN)
parser.add_argument('--arkdta_posweighted',        default=10.0,       type=float)     
parser.add_argument('--arkdta_topk_pool',          default=0.5,        type=float)
# parser.add_argument('--arkdta_contrastive',        default='None',   type=str)

args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_wandb(args):
    wandb_init = dict()
    wandb_init['project'] = args.project_name
    wandb_init['group'] = args.session_name
    if not args.testing_mode:
        wandb_init['name'] = f'training_{args.dataset_subsets}' 
    else:
        wandb_init['name'] = f'testing_{args.dataset_subsets}'
    wandb_init['notes'] = args.session_name
    os.environ['WANDB_START_METHOD'] = 'thread'

    return wandb_init

def reset_wandb_env():
    exclude = {'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_API_KEY',}
    for k, v in os.environ.items():
        if k.startswith('WANDB_') and k not in exclude:
            del os.environ[k]

def load_dataset_collate_model(args):
    if args.pred_model == 'arkdta':
        from dti.dataloaders.ArkDTA import DtiDatasetPreload, DtiDatasetAutoload
        from dti.dataloaders.ArkDTA import collate_fn
        from dti.models.ArkDTA import Net

    dataset = DtiDatasetAutoload(args) if args.dataset_loading == 'autoload' else DtiDatasetPreload(args)
    net = Net(args)

    if args.dataset_partition == 'random': dataset.make_random_splits()
    elif args.dataset_partition == 'ligand': dataset.load_predefined_splits()
    elif args.dataset_partition == 'randomsingle': dataset.make_random_splits_1fold()
    else: raise

    return dataset, collate_fn, net

def load_pretrained_model(args, net):
    if args.finetune_name != 'None':
        session_name = f'{args.project_name}_{args.session_name}'
        original_path = os.path.join(args.checkpoint_path, session_name)
        CHECKPOINT_PATH = f'{original_path}_fold_{args.fold_num}_mea_{args.ba_measure}' 
        model_config    = f'{CHECKPOINT_PATH}/model_config.pkl'
        best_model      = f'{CHECKPOINT_PATH}/best_epoch.mdl'
        last_model      = f'{CHECKPOINT_PATH}/last_epoch.mdl'
        assert os.path.isfile(model_config), f"{model_config} DOES NOT EXIST!"
        assert os.path.isfile(best_model),   f"{best_model} DOES NOT EXIST!"

        net.load_state_dict(torch.load(best_model))
        print("Loaded Pretrained Model from ", CHECKPOINT_PATH)
        args.checkpoint_path = args.checkpoint_path[:-1] + '_' + args.finetune_name + '/'
        print("ADJUSTING HYPERPARMETERS RELATED TO ARKDTA")
        torch.cuda.empty_cache()

    return args, net

def test_code(args, dataset, collate_fn, net):
    setup_seed(args.random_seed)
    setproctitle.setproctitle(f'{args.pred_model}_fold_{args.fold_num}_debug')

    session_name = 'testproject_testgroup_testsession'
    args.checkpoint_path = os.path.join(args.checkpoint_path, session_name)

    # Distributed DataLoaders
    ddp_batch_size = int(args.hp_batch_size/ngpus_per_node)
    samplers = [SubsetRandomSampler(x) for x in dataset.kfold_splits[args.fold_num-1]]

    train      = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[0], collate_fn=collate_fn)
    valid      = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[1], collate_fn=collate_fn)
    test       = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[2], collate_fn=collate_fn)
    hard_valid = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[3], collate_fn=collate_fn)
    hard_test  = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[4], collate_fn=collate_fn)

    rank = 0
    trainer = Trainer(args, rank, wandb_run=None)

    # train, valid, test and save the model
    trained_model, train_loss, valid_loss = trainer.train_valid(net, train, None, valid, hard_valid)
    if rank == 0: print('Finish Debugging Mode')

def run_single_fold(rank, ngpus_per_node, args, dataset, collate_fn, net):
    setup_seed(args.random_seed)
    pid = getpid()
    print(f'Running Process with PID: {pid}')

    setproctitle.setproctitle(f'{args.pred_model}_fold_{args.fold_num}_gpu_{rank}')
    session_name = f'{args.project_name}_{args.session_name}'
    args.checkpoint_path = os.path.join(args.checkpoint_path, session_name)

    # WANDB setup /// args
    if rank == 0:
        reset_wandb_env()
        wandb_init = setup_wandb(args)
        wandb_init['name'] += f'_{args.pred_model}_fold_{args.fold_num}_mea_{args.ba_measure}'
        run = wandb.init(**wandb_init, settings=wandb.Settings(start_method='thread'))
        run.define_metric('train/step'); run.define_metric('train/*', step_metric='train/step')
        run.define_metric('valid/step'); run.define_metric('valid/*', step_metric='valid/step')
        run.define_metric('hardv/step'); run.define_metric('hardv/*', step_metric='hardv/step')
        run.define_metric('test/step'); run.define_metric('test/*', step_metric='test/step')
        run.define_metric('hardt/step'); run.define_metric('hardt/*', step_metric='hardt/step')
        run.watch(net, log="gradients", log_freq=1000)
    else: run = None

    # initailize pytorch distributed
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', 
            init_method=f'tcp://localhost:{args.port}',
            rank=rank, world_size=ngpus_per_node)

    trainer = Trainer(args, rank, run) if args.pred_model != 'arkdta' else ArkTrainer(args, rank, run)
    trainer = net.set_default_hp(trainer) if args.hp_default_setting else trainer

    if args.hp_default_setting:
        ddp_batch_size = trainer.batch_size // ngpus_per_node
    else:
        ddp_batch_size = args.hp_batch_size // ngpus_per_node
    if rank == 0:
        print('Batch size', args.hp_batch_size)
        print('Distributed batch size', ddp_batch_size)

    dataloaders, train_sampler = [], None

    for idx, indices in enumerate(dataset.kfold_splits[args.fold_num-1]):
        sampler = DistributedSampler(Subset(dataset,indices), shuffle=True)
        loader  = DataLoader(Subset(dataset,indices), batch_size=ddp_batch_size, sampler=sampler, collate_fn=collate_fn)
        dataloaders.append(loader)
        if idx == 0: train_sampler = sampler
    del dataset
    train, valid, test, hard_valid, hard_test = dataloaders

    if args.toy_test: 
        print("Toy Test Mode"); trainer.num_epochs = 2
    if rank == 0 and not args.testing_mode:
        pickle.dump(args, open(f'{trainer.checkpoint_path}/model_config.pkl', 'wb'))

    if not args.testing_mode:
        net, train_loss, valid_loss = trainer.train_valid(net, train, train_sampler, valid, hard_valid)
        print(train_loss, valid_loss)
    else:
        chkpt = torch.load(f'{trainer.checkpoint_path}/best_epoch.mdl', map_location=f"cuda:{rank}")
        net.load_state_dict(chkpt)
        test_loss = trainer.test(net, test, hard_test)
        print(test_loss)
    print(net)

    print(f'FINISHED: {args.pred_model}_fold_{args.fold_num}_gpu_{rank}')
    time.sleep(10)
    kill(pid, SIGKILL)

def run_single_fold_multi_gpu(ngpus_per_node, args, dataset, collate_fn, net):
    torch.multiprocessing.spawn(run_single_fold, 
                                args=(ngpus_per_node, args, dataset, collate_fn, net), 
                                nprocs=ngpus_per_node, 
                                join=True)
    print("Finished Multiprocessing")


def setup_gpu(args):
    if torch.cuda.is_available():
        gpu_available = os.environ['CUDA_VISIBLE_DEVICES']
        device = f'cuda: {gpu_available}'
    else:
        device = 'cpu'

    print(f'The current world has {ngpus_per_node} GPUs?')
    print(f'The number of available CPUs is {args.num_cpus}')
    print(f'Current device is {device}\n')
    
    return args

if __name__ == "__main__":
    setup_seed(args.random_seed)
    wandb_init = setup_wandb(args)
    dataset, collate_fn, net = load_dataset_collate_model(args)
    args, net                = load_pretrained_model(args, net)
    torch.cuda.empty_cache()

    if args.debug_mode: 
        project_name = wandb_init['project']
        session_name = 'debug'
    else:
        args = setup_gpu(args)
        project_name = args.project_name
        session_name = args.session_name

    if args.debug_mode:
        test_code(args, dataset, collate_fn, net)
    else:
        run_single_fold_multi_gpu(ngpus_per_node, args, dataset, collate_fn, net)
