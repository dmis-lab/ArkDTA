import matplotlib as mpl
mpl.use('Agg')

import random
import argparse
import tarfile
import wandb
import pickle
import setproctitle
import os 
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

from rdkit import Chem 
from rdkit import RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

import torch
from torch.multiprocessing import Pool, Process, set_start_method
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns; sns.set_theme()
from matplotlib import cm
from scipy.stats import rankdata

def load_dataset_model(args):
    assert args.pred_model == 'arkdta'
    from src.dti.dataloaders.ArkDTA import DtiDatasetInfer, collate_fn
    from src.dti.models.ArkDTA import Net

    dataset = DtiDatasetInfer(args)
    net = Net(args)

    return dataset, net, collate_fn

def get_idx2smiles(mol_path, smiles):
    idx2smiles = dict()
    info = dict()
    try: mol = pickle.load(open(mol_path, 'rb'))
    except: mol = Chem.MolFromSmiles(smiles)
    try: Chem.SanitizeMol(mol)
    except: pass
    fp = AllChem.GetMorganFingerprint(mol, 2, bitInfo=info, useChirality=False)
    assert len(info) == len(fp.GetNonzeroElements())
    
    for k, v in info.items():
        for a, N in v:
            amap = dict()
            env  = Chem.FindAtomEnvironmentOfRadiusN(mol, N, a)
            submol = Chem.PathToSubmol(mol, env, atomMap=amap, useQuery=True)
            subsmiles = Chem.MolToSmiles(submol)
            if len(subsmiles) > 0: 
                idx2smiles[k%1024] = subsmiles
            else:
                atom = mol.GetAtomWithIdx(a)
                idx2smiles[k%1024] = f'{atom.GetSymbol()} ({atom.GetDegree()}d.{atom.GetExplicitValence()}v.{atom.GetFormalCharge()}c)'  
        
    return idx2smiles


class Pipeline(object):
    def __init__(self, args):
        # Load Model Arguments (Basic)
        CONFIG_PATH = args.checkpoint_path+args.project_name+'_'+args.session_name
        CONFIG_PATH = CONFIG_PATH + '_fold_'+str(args.fold_num)+'_mea_'+args.ba_measure +'/model_config.pkl'
        self.model_config = pickle.load(open(CONFIG_PATH, 'rb'))
        print(self.model_config)
        if not hasattr(self.model_config, 'deepcbal_residue_addon'):
            self.model_config.deepcbal_residue_addon = 'None'
        assert self.model_config.ba_measure == args.ba_measure

        # Load Dataset and Model
        self.model_config.root_path             = args.root_path
        self.model_config.dataset_subsets       = args.external_set
        self.model_config.debug_mode            = False
        self.model_config.toy_test              = False
        self.model_config.analysis_mode         = True
        self.dataset, self.net, self.collate_fn = load_dataset_model(self.model_config)
        self.net                                = self.net.cuda()
        self.net.eval()

        # Designate Saved Models path
        self.model_path    = args.checkpoint_path+args.project_name+'_'+args.session_name
        self.model_path    = self.model_path + '_fold_{}_mea_' + args.ba_measure + '/best_epoch.mdl'

        self.net.load_state_dict(torch.load(self.model_path.format(args.fold_num)))

    def prepare_inputdata(self, sequence_pair):
        fasta       = sequence_pair[0]
        smiles      = sequence_pair[1]
        complex_idx = sequence_pair[2]

        mol = Chem.MolFromSmiles(smiles)
        try: Chem.SanitizeMol(mol)
        except: pass 

        ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=1024, useChirality=False))
        ecfp_words = np.nonzero(ecfp)[0].reshape(1,-1)
        resiwise_features = np.vstack([np.zeros(21).reshape(1, -1) for _ in fasta]).astype(int) 
        atomresi_graph =  np.zeros((len(mol.GetAtoms()), len(fasta),1))

        return self.collate_fn([(ecfp_words,resiwise_features,fasta,atomresi_graph,0,False,complex_idx)])

    @torch.no_grad()
    def __call__(self, complex_idx, sequence_pair):

        if complex_idx:
            assert sequence_pair == None
            result = self.net(self.dataset.get_test_instance(complex_idx))
            return result['task/ba_pred'].item(), result['task/ba_true'].item()

        if sequence_pair:            
            assert complex_idx == None
            result = self.net(self.prepare_inputdata(sequence_pair))
            return result['task/ba_pred'].item(), result['task/ba_true'].item()

    def prepare_heatmap_attention_weights(self, complex_idx=None, sequence_pair=None):
        if complex_idx:
            assert sequence_pair == None
            input_data  = self.dataset.get_test_instance(complex_idx)
            prot_fastas = input_data[1][0][1]
            comp_substr = input_data[3].view(-1)

            ligand_idx = self.dataset.complex_dataframe.loc[complex_idx, 'ligand_id']
            mol_path = f'{self.dataset.data_path}ligands/{ligand_idx}/{ligand_idx}.mol'
            smiles = self.dataset.ligand_dataframe.loc[ligand_idx, 'smiles']
            idx2smiles = get_idx2smiles(mol_path, smiles)

            result = self.net(input_data)
            ba_pred, ba_true = result['task/ba_pred'].item(), result['task/ba_true'].item()
            attn_weights = self.net.layer['intg_arkmab'].pmx.mab.multihead.attention.attention_maps[0]
            self.net.layer['intg_arkmab'].pmx.mab.multihead.attention.attention_maps = []
            attn_mean = attn_weights.mean(0)
            attn_list = [attn_weights[i,:,:] for i in range(attn_weights.shape[0])]

            plip_path = f'{self.dataset.data_path}complexes/{complex_idx}/{complex_idx}.plip.npy'
            try: noncov_labels = 1. - np.load(plip_path)[:,:,-1]
            except: noncov_labels = np.zeros((len(mol.GetAtoms()), len(prot_fastas)))

            return prot_fastas, comp_substr, idx2smiles, noncov_labels, attn_mean, attn_list, complex_idx, ba_pred, ba_true

        if sequence_pair:
            assert complex_idx == None
            input_data = self.prepare_inputdata(sequence_pair)
            result = self.net(input_data)
            ba_pred, ba_true = result['task/ba_pred'].item(), result['task/ba_true'].item()

            attn_weights = self.net.layer['intg_arkmab'].pmx.mab.multihead.attention.attention_maps[0]
            self.net.layer['intg_arkmab'].pmx.mab.multihead.attention.attention_maps = []
            attn_mean = attn_weights.mean(0)
            attn_list = [attn_weights[i,:,:] for i in range(attn_weights.shape[0])]

            prot_fastas = sequence_pair[0]
            smiles      = sequence_pair[1]
            complex_idx = sequence_pair[2]
            idx2smiles = get_idx2smiles('None', smiles)
            comp_substr = input_data[3].view(-1)
            mol = Chem.MolFromSmiles(smiles)
            try: Chem.SanitizeMol(mol)
            except: pass 
            noncov_labels = np.zeros((len(mol.GetAtoms()), len(prot_fastas)))

            return prot_fastas, comp_substr, idx2smiles, noncov_labels, attn_mean, attn_list, complex_idx, ba_pred, ba_true
            
    def visualize_heatmap_attention_weights(self, prot_fastas, comp_substr, 
                                                  idx2smiles, noncov_labels, 
                                                  attn_mean, attn_list, 
                                                  complex_idx, ba_pred, ba_true):
        return_figs = {'mean': None}

        def value2rank(X):
            return X.shape[1] - rankdata(X, 'min', axis=1).astype(int)

        residue_labels = list(prot_fastas)
        compsub_labels = comp_substr.view(-1).cpu().numpy().tolist()
        compsub_labels = [idx2smiles[x] for x in compsub_labels]
        compsub_labels.append('Predicted NCI Score')

        noncov_labels = (noncov_labels.sum(0) > 0.).astype(np.float_).reshape(-1,1)
        noncov_labels[noncov_labels==1] = np.nan
        noncov_labels[noncov_labels==0] = -1

        def f(title, data, compsub_labels, residue_labels, noncov_labels):
            plt.tight_layout()
            plt.figure(figsize=(data.shape[1]*1.2, data.shape[0]*1.2))
            plt.title(title, fontsize=20)
            plt.rcParams['xtick.bottom'] = False
            plt.rcParams['xtick.labelbottom'] = False
            plt.rcParams['xtick.top'] = True
            plt.rcParams['xtick.labeltop'] = True
 
            # Attention Weights when Non-Covalent Interaction is Predicted as 1
            data0 = copy.deepcopy(data)
            data0[:,-1] = -1
            cmap0 = copy.copy(plt.get_cmap("YlGn"))
            cmap0.set_under("none")
            cmap0.set_bad("black")

            # Attention Weights when Non-Covalent Interaction is Predicted as 0
            data1 = copy.deepcopy(data)
            data1[:,:-1] = -1
            cmap1 = copy.copy(plt.get_cmap("Reds_r"))            
            cmap1.set_under("none")
            cmap1.set_bad("black")

            hm_args0 = {'data': np.hstack([data0,noncov_labels]),
                        'square': True, 'cbar': True, 'cmap': cmap0, 
                        'vmin': 0.0,
                        'xticklabels':compsub_labels+['Actual NCI Label'],
                        'yticklabels':residue_labels,
                        'cbar_kws': {"orientation":"horizontal"}
                        }
            sns.set(font_scale=3.0)
            ax = sns.heatmap(**hm_args0)

            hm_args1 = {'data': np.hstack([data1,noncov_labels]), 'ax':ax,
                        'square': True, 'cbar': True, 'cmap': cmap1, 
                        'vmin': 0.0, 'vmax': 1.0, 
                        'linewidths': 0.01, 'linecolor': 'black',
                        'xticklabels':compsub_labels+['Actual NCI Label'],
                        'yticklabels':residue_labels,
                        'cbar_kws': {"orientation":"horizontal", }
                        }
            sns.set(font_scale=3.0)
            return sns.heatmap(**hm_args1)

        return_figs['mean'] = f(complex_idx + ' MEAN', attn_mean, 
                                compsub_labels, residue_labels, noncov_labels)
        for idx, am in enumerate(attn_list):
            return_figs[f'head{idx+1}'] = f(complex_idx + f' HEAD #{idx+1}', am, 
                                            compsub_labels, residue_labels, noncov_labels)
        return_figs['mean'] = f(complex_idx + ' MEAN', attn_mean, 
                                compsub_labels, residue_labels, noncov_labels)

        if not os.path.exists('./plots'):
            os.makedirs('./plots')
        for k,v in return_figs.items():
            png_name = f'./plots/{complex_idx}_{k}_pred{ba_pred:.3f}_true{ba_true:.3f}.png'
            v.get_figure().savefig(png_name)
            print(f"SAVED FIGURE: {png_name}")

        return return_figs

    def analyze_heatmap_attention_weights(self, complex_idx=None, sequence_pair=None):
        input_args = self.prepare_heatmap_attention_weights(complex_idx, sequence_pair)
        self.visualize_heatmap_attention_weights(*input_args)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',       '-rp', default='../', type=str)
    parser.add_argument('--checkpoint_path', '-cp', default='./saved/', type=str)
    parser.add_argument('--project_name',    '-pn', default='ISMB2023_ArkDTA', type=str)
    parser.add_argument('--session_name',    '-sn', default='arkdta', type=str)
    parser.add_argument('--external_set',    '-ex', default='pdb_2020_general', type=str)
    parser.add_argument('--ba_measure',      '-ba', default='KIKD', type=str)
    parser.add_argument('--disable_wandb',   '-dw', default=True)
    parser.add_argument('--fold_num',        '-fn', default=3, type=int)

    parser.add_argument('--input_type',      '-it', default='pdb', type=str, choices=['pdb', 'dtp'])
    parser.add_argument('--analysis_id',     '-id', default='None', type=str) # pdb_2020_generalBC016326
    parser.add_argument('--analysis_smiles', '-sm', default=None, type=str)
    parser.add_argument('--analysis_fasta' , '-fa', default=None, type=str)
    parser.add_argument('--analysis_option', '-ao', default=None, type=str, choices=[None, 'hm_aw'])
    args = parser.parse_args()
    
    if args.input_type == 'dtp':
        sequence_pair = (args.analysis_fasta, args.analysis_smiles, args.analysis_id)
        complex_idx   = None
    elif args.input_type == 'pdb':
        sequence_pair = None
        complex_idx   = args.analysis_id
    else:
        raise

    if args.analysis_option:
        pipeline = Pipeline(args)
        if args.analysis_option == 'hm_aw':
            input_args = pipeline.prepare_heatmap_attention_weights(complex_idx, sequence_pair)
            pipeline.visualize_heatmap_attention_weights(*input_args)

        elif args.analysis_option == 'hm_sv':

            pipeline.test_function(complex_idx, sequence_pair)

# python analysis.py -pn ISMB2023_ArkDTA -sn arkdta -ao hm_aw -ex pdb_2020_general -it dtp -id 8BQ4_QZR -sm 'Cc1cc2c(ncnc2s1)Nc3ccc(cc3)S(=O)(=O)C' -fa DPLVGVFLWGVAHSINELSQVPPPVMLLPDDFKASSKIKVNNHLFHRENLPSHFKFKEYCPQVFRNLRDRFGIDDQDYLVSLTRNPPSESEGSDGRFLISYDRTLVIKEVSSEDIADMHSNLSNYHQYIVKCHGNTLLPQFLGMYRVSVDNEDSYMLVMRNMFSHRLPVHRKYDLKGSLVSREASDKEKVKELPTLKDMDFLNKNQKVYIGEEEKKIFLEKLKRDVEFLVQLKIMDYSLLLGIHDIIRGSEPEEELGPGEFESFIDVYAIRSAEGAPQKEVYFMGLIDILTQYDAKKKAAHAAKTVKHGAGAEISTVHPEQYAKRFLDFITNIFA