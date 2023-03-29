import pickle
import pandas as pd
import numpy as np
from rdkit import Chem 
from rdkit import RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
import os
import sys
import torch
from torch.utils.data import Dataset 
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
import time
import itertools
from collections import defaultdict
from gensim.models import Word2Vec
import math
from rdkit.Chem.Scaffolds.MurckoScaffold import *

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))

###############################################
#                                             #
#              Dataset Base Class             #
#                                             #
###############################################


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def onek_encoding(x, allowable_set):
    if x not in allowable_set:                                                                                                                                               
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def featurization(x):

    return x

def check_exists(path):    
    return True if os.path.isfile(path) and os.path.getsize(path) > 0 else False

def add_index(input_array, ebd_size):
    add_idx, temp_arrays = 0, []
    for i in range(input_array.shape[0]):
        temp_array = input_array[i,:,:]
        masking_indices = temp_array.sum(1).nonzero()
        temp_array += add_idx
        temp_arrays.append(temp_array)
        add_idx = masking_indices[0].max()+1
    new_array = np.concatenate(temp_arrays, 0)

    return new_array.reshape(-1)

def get_substruct_smiles(smiles):
    tic = time.perf_counter()
    info = dict()
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    fp = AllChem.GetMorganFingerprint(mol, 2, bitInfo=info)
    ec = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,useChirality=False))
    X = []

    for k, v in info.items():
        for a, N in v:
            amap = dict()
            env  = Chem.FindAtomEnvironmentOfRadiusN(mol, N, a)
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            subsmiles = Chem.MolToSmiles(submol)
            if len(subsmiles) > 0: X.append(subsmiles)
        
    X = list(set(X))
    # print(f"Code Execution: {toc - tic:0.4f} seconds")
    return X

class DtiDatasetBase(Dataset):
    def __init__(self, args):
        self.args = args
        self.data_instances, self.meta_instances = [], []
        self.analysis_mode = args.analysis_mode
        self.num_subsample = args.dataset_subsample

        # Gathering All Meta-data from DTI Datasets
        self.data_path = os.path.join(args.root_path, f'dataset_{args.dataset_version}/')
        complex_dataframe, protein_dataframe, ligand_dataframe = [], [], []
        self.list_datasets = []
        
        # args.dataset_subsets = 'pdb_2020_general+kiba+davis+metz+bddb+human+pcba+celegans'
        for dataset in args.dataset_subsets.split('+'):
            complex_path = f'{self.data_path}complex_metadata_{dataset}.csv'
            protein_path = f'{self.data_path}protein_metadata_{dataset}.csv'
            ligand_path = f'{self.data_path}ligand_metadata_{dataset}.csv'
            complex_dataframe.append(pd.read_csv(complex_path, index_col='complex_id'))
            protein_dataframe.append(pd.read_csv(protein_path, index_col='protein_id'))
            ligand_dataframe.append(pd.read_csv(ligand_path, index_col='ligand_id'))

            # if dataset == 'davis':
            #     temp = complex_dataframe[-1]
            #     complex_dataframe[-1] = temp[temp['ba_value']>5.0]
            
            if dataset == 'bddb':
                temp = complex_dataframe[-1]
                temp = temp.replace([np.inf, -np.inf], np.nan).dropna(subset=['ba_value'], axis=0)
                temp = temp[temp.ba_value>=2.0]
                complex_dataframe[-1] = temp[temp.ba_value<=14.0]
                # complex_dataframe[-1] = temp[temp.ba_value>0.0]

            self.list_datasets.append(dataset)
        
        self.complex_dataframe = pd.concat(complex_dataframe)
        self.protein_dataframe = pd.concat(protein_dataframe)
        self.ligand_dataframe = pd.concat(ligand_dataframe)

        def ic50_threshold(x):
            if float(x['ba_value']) > 7.0: return 1
            elif float(x['ba_value']) < 5.0: return 0 
            else: return np.nan

        def kikd_threshold(x):
            if float(x['ba_value']) > 7.0: return 1
            else: return 0

        # def kikd_threshold(x):
        #     if float(x['ba_value']) > 7.0: return 1
        #     else: return 0

        if args.dataset_subsets in ['bddb', 'pdb_2020_general', 'pdb_2020_refined']:
            ic50 = self.complex_dataframe[self.complex_dataframe['ba_measure']=='IC50']
            kikd = self.complex_dataframe[self.complex_dataframe['ba_measure']=='KIKD']
            ic50['ba_label'] = ic50.apply(lambda x: ic50_threshold(x), axis=1)
            kikd['ba_label'] = kikd.apply(lambda x: kikd_threshold(x), axis=1)
            self.complex_dataframe = pd.concat([ic50,kikd])

        if args.dataset_subsets in ['davis', 'metz']:
            self.complex_dataframe['ba_label'] = self.complex_dataframe.apply(lambda x: kikd_threshold(x), axis=1)

        if args.dataset_subsets in ['human', 'pcba', 'celegans']:
            self.complex_dataframe['ba_label'] = self.complex_dataframe['ba_value']

        if args.ba_measure == 'Binary':
            self.complex_dataframe['ba_measure'] = 'Binary'

        self.complex_dataframe = self.complex_dataframe[self.complex_dataframe['ba_measure']==args.ba_measure]
        self.complex_dataframe.dropna(subset=['ba_value'], axis=0, inplace=True)
        self.complex_dataframe.dropna(subset=['ba_label'], axis=0, inplace=True)

        self.complex_indices = self.complex_dataframe.index
        if args.debug_mode or args.toy_test:
            self.complex_indices = self.complex_dataframe.sample(n=args.debug_index).index
        if self.num_subsample:
            self.complex_dataframe = self.complex_dataframe.sample(n=self.num_subsample)

        self.kfold_splits = []

    def check_ligand(self, ligand_idx):
        return

    def check_protein(self, protein_idx):
        if self.protein_dataframe.loc[protein_idx, 'fasta_length'] >= 1000:
            raise FastaLengthException(self.protein_dataframe.loc[protein_idx, 'fasta_length'])

    def check_complex(self, complex_idx):
        return 

    def __len__(self):
        return len(self.data_instances)

    def __getitem__(self, idx):
        if self.analysis_mode:
            return self.data_instances[idx], self.meta_instances[idx]
        else:
            return self.data_instances[idx]

    def get_scaffold_smiles(self, ligand_idx, smiles):
        mol_path = f'{self.data_path}ligands/{ligand_idx}/{ligand_idx}.mol'
        try:
            mol = pickle.load(open(mol_path, 'rb'))
            try: Chem.SanitizeMol(mol)
            except: pass
            scaffold = MurckoScaffoldSmiles(mol=mol)
        except:
            try: scaffold = MurckoScaffoldSmilesFromSmiles(smiles)
            except: scaffold = ''

        return scaffold if len(scaffold) > 0 else 'unknown'

    def label_hard_samples(self, train_indices, valid_indices, test_indices):
        print("Labeling Hard Samples")
        train_scaffolds = []
        hard_valid_indices, hard_test_indices = [], []
        for train_idx in tqdm(train_indices):
            meta_instance = self.meta_instances[train_idx]
            train_scaffolds.append(self.get_scaffold_smiles(meta_instance[1], meta_instance[3]))
        train_scaffolds = set(train_scaffolds)
        if 'unknown' in train_scaffolds: 
            train_scaffolds.remove('unknown')

        for valid_idx in tqdm(valid_indices):
            meta_instance = self.meta_instances[valid_idx]
            smiles = self.get_scaffold_smiles(meta_instance[1], meta_instance[3])
            if smiles not in train_scaffolds: 
                hard_valid_indices.append(valid_idx)

        for test_idx in tqdm(test_indices):
            meta_instance = self.meta_instances[test_idx]
            smiles = self.get_scaffold_smiles(meta_instance[1], meta_instance[3])
            if smiles not in train_scaffolds: 
                hard_test_indices.append(test_idx)  

        print("Number of Training Samples",   len(train_indices))    
        print("Number of Validation Samples", len(valid_indices))
        print("Number of Test Samples",       len(test_indices))
        print("Number of Hard Val Samples",   len(hard_valid_indices))
        print("Number of Hard Test Samples",  len(hard_test_indices))

        if len(hard_valid_indices) == 0:
            hard_valid_indices = valid_indices
        if len(hard_test_indices) == 0:
            hard_test_indices = test_indices

        return np.array(hard_valid_indices), np.array(hard_test_indices)

    def make_random_splits(self):
        print("Making Random Splits")
        kf = KFold(n_splits=5, shuffle=True)
        for train_indices, test_indices in kf.split(self.indices):
            train_indices, valid_indices = train_test_split(train_indices, test_size=0.05)
            assert len(set(train_indices) & set(valid_indices)) == 0
            assert len(set(valid_indices) & set(test_indices)) == 0
            assert len(set(train_indices) & set(test_indices)) == 0
            hard_valid, hard_test = self.label_hard_samples(train_indices, valid_indices, test_indices)
            self.kfold_splits.append((train_indices.tolist(), 
                                      valid_indices.tolist(), test_indices.tolist(), 
                                      hard_valid.tolist(), hard_test.tolist()))

    def load_predefined_splits(self):
        import json
        print("Loading Predefined Splits")
        assert self.args.dataset_subsets == 'pdb_2020_refined' or self.args.dataset_subsets == 'pdb_2020_general'
        split_path = f'{self.data_path}ligand_{self.args.dataset_subsets}_{self.args.ba_measure}.json'
        split_indices = json.load(open(split_path))
        for i in range(5):
            train_indices, valid_indices, test_indices = [], [], []
            for idx, meta_instance in enumerate(self.meta_instances):
                ligand_idx = meta_instance[1]
                if ligand_idx in split_indices[str(i+1)]['train']: train_indices.append(idx)
                elif ligand_idx in split_indices[str(i+1)]['valid']: valid_indices.append(idx)
                elif ligand_idx in split_indices[str(i+1)]['test']: test_indices.append(idx)
                else: pass
            hard_valid, hard_test = self.label_hard_samples(train_indices, valid_indices, test_indices)
            self.kfold_splits.append((train_indices, valid_indices, test_indices, hard_valid, hard_test))

    def make_random_splits_1fold(self):
        if self.num_subsample:
            print("Subsampling...")
            self.indices = np.random.choice(self.indices, min(self.num_subsample, len(self.indices))).tolist()
        print("Making Random Splits without 5CV")
        train_indices, test_indices  = train_test_split(self.indices, test_size=0.2)
        train_indices, valid_indices = train_test_split(train_indices, test_size=0.05)
        hard_valid, hard_test = self.label_hard_samples(train_indices, valid_indices, test_indices)
        for i in range(5): 
            self.kfold_splits.append((train_indices, valid_indices, test_indices, hard_valid, hard_test))

    def minimize(self, sub_idx):
        minimized_indices =  list(itertools.chain(*self.kfold_splits[sub_idx]))
        self.meta_instances = list(map(self.meta_instances.__getitem__, minimized_indices))
        self.data_instances = list(map(self.data_instances.__getitem__, minimized_indices))

class SmilesLengthException(Exception):
    def __init__(self, smiles_length, message="smiles length should not exceed 100"):
        self.smiles_length = smiles_length
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.smiles_length} -> {self.message}'

class FastaLengthException(Exception):
    def __init__(self, fasta_length, message="fasta length should not exceed 1000"):
        self.fasta_length = fasta_length
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.fasta_length} -> {self.message}'

class NoProteinGraphException(Exception):
    def __init__(self, protein_idx, message="protein graph structure file not available"):
        self.protein_idx = protein_idx
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.protein_idx} -> {self.message}'

class NoProteinFeaturesException(Exception):
    def __init__(self, protein_idx, message="protein advanced features file not available"):
        self.protein_idx = protein_idx
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.protein_idx} -> {self.message}'

class NoComplexGraphException(Exception):
    def __init__(self, complex_idx, message="complex advanced features file not available"):
        self.complex_idx = complex_idx
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.complex_idx} -> {self.message}'

class NoInteractionException(Exception):
    def __init__(self, complex_idx, message="the non-covalent interaction is not available"):
        self.complex_idx = complex_idx
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.complex_idx} -> {self.message}'

class NullBondFeaturesException(Exception):
    def __init__(self, bond_features, message="one of the bond features is null"):
        self.bond_features = bond_features
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.bond_features} -> {self.message}'

class FastaLowerCaseException(Exception):
    def __init__(self, fasta, message="fasta letters are all lower-case"):
        self.fasta = fasta
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.fasta} -> {self.message}'

class FastaNumeralsException(Exception):
    def __init__(self, fasta, message="there are some numerals in fasta"):
        self.fasta = fasta
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.fasta} -> {self.message}'



###############################################
#                                             #
#              Collate Functions              #
#                                             #
###############################################



def stack_and_pad(arr_list, max_length=None):
    M = max([x.shape[0] for x in arr_list]) if not max_length else max_length
    N = max([x.shape[1] for x in arr_list])
    T = np.zeros((len(arr_list), M, N))
    t = np.zeros((len(arr_list), M))
    s = np.zeros((len(arr_list), M, N))

    for i, arr in enumerate(arr_list):
        # sum of 16 interaction type, one is enough
        if len(arr.shape) > 2:
            arr = (arr.sum(axis=2) > 0.0).astype(float)
        T[i, 0:arr.shape[0], 0:arr.shape[1]] = arr
        t[i, 0:arr.shape[0]] = 1 if arr.sum() != 0.0 else 0
        s[i, 0:arr.shape[0], 0:arr.shape[1]] = 1 if arr.sum() != 0.0 else 0
    return T, t, s

def stack_and_pad_with(arr_list, max_length=None, padding_idx=0):
    M = max([x.shape[0] for x in arr_list]) if not max_length else max_length
    N = max([x.shape[1] for x in arr_list])
    # T = np.zeros((len(arr_list), M, N))
    T = np.full((len(arr_list), M, N), padding_idx)
    t = np.zeros((len(arr_list), M))
    s = np.zeros((len(arr_list), M, N))

    for i, arr in enumerate(arr_list):
        # sum of 16 interaction type, one is enough
        if len(arr.shape) > 2:
            arr = (arr.sum(axis=2) > 0.0).astype(float)
        T[i, 0:arr.shape[0], 0:arr.shape[1]] = arr
        t[i, 0:arr.shape[0]] = 1 if arr.sum() != 0.0 else 0
        s[i, 0:arr.shape[0], 0:arr.shape[1]] = 1 if arr.sum() != 0.0 else 0
    return T, t, s

def stack_and_pad_2d(arr_list, block='lower_left', max_length=None):
    max0 = max([a.shape[0] for a in arr_list]) if not max_length else max_length
    max1 = max([a.shape[1] for a in arr_list])
    list_shapes = [a.shape for a in arr_list]

    final_result = np.zeros((len(arr_list), max0, max1))
    final_masks_2d = np.zeros((len(arr_list), max0))
    final_masks_3d = np.zeros((len(arr_list), max0, max1))

    if block == 'upper_left':
        for i, shape in enumerate(list_shapes):
            # sum of 16 interaction type, one is enough
            if len(arr_list[i].shape) > 2:
                arr_list[i] = (arr_list[i].sum(axis=2) == True).astype(float)
            final_result[i, :shape[0], :shape[1]] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], :shape[1]] = 1
    elif block == 'lower_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, max1-shape[1]:] = 1
    elif block == 'lower_left':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, :shape[1]] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, :shape[1]] = 1
    elif block == 'upper_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, :shape[0], max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], max1-shape[1]:] = 1
    else:
        raise

    return final_result, final_masks_2d, final_masks_3d

def stack_and_pad_3d(arr_list, block='lower_left'):
    max0 = max([a.shape[0] for a in arr_list])
    max1 = max([a.shape[1] for a in arr_list])
    max2 = max([a.shape[2] for a in arr_list])
    list_shapes = [a.shape for a in arr_list]

    final_result = np.zeros((len(arr_list), max0, max1, max2))
    final_masks_2d = np.zeros((len(arr_list), max0))
    final_masks_3d = np.zeros((len(arr_list), max0, max1))
    final_masks_4d = np.zeros((len(arr_list), max0, max1, max2))

    if block == 'upper_left':
        for i, shape in enumerate(list_shapes):
            final_result[i, :shape[0], :shape[1], :shape[2]] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], :shape[1]] = 1
            final_masks_4d[i, :shape[0], :shape[1], :] = 1
    elif block == 'lower_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, max1-shape[1]:] = 1
            final_masks_4d[i, max0-shape[0]:, max1-shape[1]:, :] = 1
    elif block == 'lower_left':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, :shape[1]] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, :shape[1]] = 1
            final_masks_4d[i, max0-shape[0]:, :shape[1], :] = 1
    elif block == 'upper_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, :shape[0], max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], max1-shape[1]:] = 1
            final_masks_4d[i, :shape[0], max1-shape[1]:, :] = 1
    else:
        raise

    return final_result, final_masks_2d, final_masks_3d, final_masks_4d

def ds_normalize(input_array):
    # Doubly Stochastic Normalization of Edges from CVPR 2019 Paper
    assert len(input_array.shape) == 3
    input_array = input_array / np.expand_dims(input_array.sum(1)+1e-8, axis=1)
    output_array = np.einsum('ijb,jkb->ikb', input_array,
                             input_array.transpose(1, 0, 2))
    output_array = output_array / (output_array.sum(0)+1e-8)

    return output_array