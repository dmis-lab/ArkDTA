import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import yaml
from yaml.loader import SafeLoader
import argparse
from datetime import datetime
now = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--root_path',        '-rn', default='../',      type=str)
parser.add_argument('--checkpoint_path',  '-cn', default='./saved/', type=str)

parser.add_argument('--port',             '-pt', default=12349,      type=int)
parser.add_argument('--debug_mode',       '-dm', default=False,      action='store_true')
parser.add_argument('--toy_test',         '-tt', default=False,      action='store_true')
parser.add_argument('--multi_gpu',        '-mg', default='0,1',      type=str)

parser.add_argument('--project_name',     '-pn', default='ISMB2023', type=str)
parser.add_argument('--session_name',     '-sn', default='v0',       type=str)
parser.add_argument('--finetune_name',    '-ft', default=None,       type=str)

parser.add_argument('--multi_fold',       '-mf', default=1,          type=int)
parser.add_argument('--start_fold',       '-sf', default=1,          type=int)
parser.add_argument('--end_fold',         '-ef', default=5,          type=int)
parser.add_argument('--testing_mode',     '-tm', default=False,      action='store_true')
parser.add_argument('--baseline_default', '-bd', default=False,      action='store_true')

args = parser.parse_args()

with open(f'sessions/{args.session_name}.yaml') as f:
    arg_dict = dict(yaml.load(f, Loader=SafeLoader))

SCRIPT_LINE = f'CUDA_VISIBLE_DEVICES={args.multi_gpu} python -W ignore src/train.py'
for k,v in arg_dict.items():
    SCRIPT_LINE += f' --{k} {v}'
SCRIPT_LINE += f' --project_name {args.project_name}'
SCRIPT_LINE += f' --root_path {args.root_path}'
SCRIPT_LINE += f' --session_name {args.session_name}' 
SCRIPT_LINE += f' --checkpoint_path {args.checkpoint_path}'
if args.debug_mode:       SCRIPT_LINE += ' --debug_mode'
if args.toy_test:         SCRIPT_LINE += ' --toy_test'
if args.testing_mode:     SCRIPT_LINE += ' --testing_mode'
if args.baseline_default: SCRIPT_LINE += ' --hp_default_setting'
if args.finetune_name:    SCRIPT_LINE += f' --finetune_name {args.finetune_name}'

def run_process(fold_num, port):
    os.system(f'{SCRIPT_LINE} --fold_num {fold_num} --port {port}')
    return fold_num

def multiprocess():
    if args.toy_test:
        print('########################### Toy Test ###########################')
        
    from multiprocessing import Pool
    pool = Pool(args.multi_fold)

    all_folds = [*range(args.start_fold, args.end_fold+1)]
    run_folds_list = [all_folds[start_fold:(start_fold+args.end_fold)]
                      for start_fold in range(0, args.end_fold, args.end_fold)]
    if args.toy_test: run_folds_list = [[1]]

    fold_results_list = []
    #run_folds_list = [[4]]
    for fold in run_folds_list:
        print('Dataset Fold Index: ', fold)
        args_list = [(fold_idx, args.port+fold_idx) for fold_idx in fold]
        fold_results_list.extend(pool.starmap(run_process, args_list))
    pool.close()
    pool.join()


if __name__ == "__main__":
    if args.debug_mode:
        print('########################### Debug Mode ###########################')
        run_process(0, args.port)
    else:
        multiprocess()
