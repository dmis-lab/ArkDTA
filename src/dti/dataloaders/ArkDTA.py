from .base import *

ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
             'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 
             'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
             'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 
             'Pt', 'Hg', 'Pb', 'unk']

AMINO_ACID_LIST = ['A', 'C', 'D', 'E', 'F', 
                   'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 
                   'S', 'T', 'V', 'W', 'Y', 'unk']


class DtiDatasetCore(DtiDatasetBase):
    def __init__(self, args):
        super().__init__(args)
        self.ecfpvec_dim = args.arkdta_ecfpvec_dim

    def check_protein(self, protein_idx):
        if self.protein_dataframe.loc[protein_idx, 'fasta_length'] >= 1000:
            raise FastaLengthException(self.protein_dataframe.loc[protein_idx, 'fasta_length'])
        if self.protein_dataframe.loc[protein_idx, 'fasta'].islower():
            raise FastaLowerCaseException(self.protein_dataframe.loc[protein_idx, 'fasta'])
        if any(char.isdigit() for char in self.protein_dataframe.loc[protein_idx, 'fasta']):
            raise FastaNumeralsException(self.protein_dataframe.loc[protein_idx, 'fasta'])

    def check_complex(self, complex_idx):
        #if not check_exists(f'{self.data_path}complexes/{complex_idx}/{complex_idx}.arpeggio.npy'):
        if not check_exists(f'{self.data_path}complexes/{complex_idx}/{complex_idx}.plip.npy'):
            raise NoComplexGraphException(complex_idx)  

    def atom_features(self, atom):
        return np.array(onek_encoding_unk(atom.GetSymbol(), ATOM_LIST), dtype=np.float32)

    def resi_features(self, resi):
        return np.array(onek_encoding_unk(resi, AMINO_ACID_LIST))    


class DtiDatasetPreload(DtiDatasetCore):
    def __init__(self, args):
        super().__init__(args)
        for complex_idx in tqdm(self.complex_indices):
            try:
                ligand_idx = self.complex_dataframe.loc[complex_idx, 'ligand_id']
                protein_idx = self.complex_dataframe.loc[complex_idx, 'protein_id']
                ba_value = self.complex_dataframe.loc[complex_idx, 'ba_value']
                ba_label = self.complex_dataframe.loc[complex_idx, 'ba_label']
                self.check_protein(protein_idx)

                # Ligand / Atom / Features
                atomwise_features = []
                mol_path = f'{self.data_path}ligands/{ligand_idx}/{ligand_idx}.mol'
                smiles = self.ligand_dataframe.loc[ligand_idx, 'smiles']
                if check_exists(mol_path): mol = pickle.load(open(mol_path, 'rb'))
                else: mol = Chem.MolFromSmiles(smiles)
                try: Chem.SanitizeMol(mol)
                except: pass 
                try:
                    ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,
                                                                          nBits=self.ecfpvec_dim,
                                                                          useChirality=False))
                except:
                    ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,
                                                                          nBits=self.ecfpvec_dim,
                                                                          useChirality=False))
                ecfp_words = np.nonzero(ecfp)[0].reshape(1,-1)  

                # Protein / Residue / Features
                resiwise_features = []
                fasta = self.protein_dataframe.loc[protein_idx, 'fasta']
                for resi in fasta:
                    try: resiwise_features.append(self.resi_features(resi).reshape(1, -1))
                    except: resiwise_features.append(np.zeros(21).reshape(1, -1))
                resiwise_features = np.vstack(resiwise_features).astype(int)    

                plip_path = f'{self.data_path}complexes/{complex_idx}/{complex_idx}.plip.npy'
                if check_exists(plip_path):
                    atomresi_graph = np.load(plip_path)[:,:,:-1]
                else:
                    atomresi_graph = np.zeros((len(mol.GetAtoms()), len(fasta), 1))
                    
                if atomresi_graph.sum() == 0 and 'pdb' in self.list_datasets: 
                    raise NoInteractionException(complex_idx)

                metadata = (complex_idx, ligand_idx, protein_idx, smiles, fasta, ba_value, ba_label)
                pytrdata = (ecfp_words, resiwise_features, fasta, atomresi_graph, ba_value, ba_label, complex_idx)

                self.data_instances.append(pytrdata)
                self.meta_instances.append(metadata)

            except Exception as e:
                pass

        self.indices = [i for i in range(len(self.data_instances))]


class DtiDatasetAutoload(DtiDatasetCore):
    def __init__(self, args):
        super().__init__(args)
        for complex_idx in tqdm(self.complex_indices):
            try:
                ligand_idx = self.complex_dataframe.loc[complex_idx, 'ligand_id']
                protein_idx = self.complex_dataframe.loc[complex_idx, 'protein_id']
                ba_value = self.complex_dataframe.loc[complex_idx, 'ba_value']
                ba_label = self.complex_dataframe.loc[complex_idx, 'ba_label']
                smiles = self.ligand_dataframe.loc[ligand_idx, 'smiles']
                fasta = self.protein_dataframe.loc[protein_idx, 'fasta']
                self.check_protein(protein_idx)

                metadata = (complex_idx, ligand_idx, protein_idx, smiles, fasta, ba_value, ba_label)
                self.meta_instances.append(metadata)

            except Exception as e:
                pass

        print("Number of data samples for ArkDTA: ", len(self.meta_instances))
        self.indices = [i for i in range(len(self.meta_instances))]
        
    def __len__(self):
        return len(self.meta_instances)

    def __getitem__(self, idx):
        complex_idx, ligand_idx, protein_idx, _, _, _, _ = self.meta_instances[idx]
        ba_value = self.complex_dataframe.loc[complex_idx, 'ba_value']
        ba_label = self.complex_dataframe.loc[complex_idx, 'ba_label']

        # Ligand / Atom / Features
        mol_path = f'{self.data_path}ligands/{ligand_idx}/{ligand_idx}.mol'
        smiles = self.ligand_dataframe.loc[ligand_idx, 'smiles']
        if check_exists(mol_path): mol = pickle.load(open(mol_path, 'rb'))
        else: mol = Chem.MolFromSmiles(smiles)
        try: Chem.SanitizeMol(mol)
        except: pass 
        try:
            ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,
                                                                  nBits=self.ecfpvec_dim,
                                                                  useChirality=False))
        except:
            ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,
                                                                  nBits=self.ecfpvec_dim,
                                                                  useChirality=False))
        ecfp_words = np.nonzero(ecfp)[0].reshape(1,-1)  

        # Protein / Residue / Features
        resiwise_features = []
        fasta = self.protein_dataframe.loc[protein_idx, 'fasta']
        for resi in fasta:
            try: resiwise_features.append(self.resi_features(resi).reshape(1, -1))
            except: resiwise_features.append(np.zeros(21).reshape(1, -1))
        resiwise_features = np.vstack(resiwise_features).astype(int)    

        plip_path = f'{self.data_path}complexes/{complex_idx}/{complex_idx}.plip.npy'
        if check_exists(plip_path):
            atomresi_graph = np.load(plip_path)[:,:,:-1]
        else:
            atomresi_graph = np.zeros((len(mol.GetAtoms()), len(fasta), 1))

        return (ecfp_words, resiwise_features, fasta, atomresi_graph, ba_value, ba_label, complex_idx)


def collate_fn(batch):
    tensor_list = []
    list_ecfp_words        = [x[0] for x in batch]
    list_resiwise_features = [x[1] for x in batch]
    list_fastas            = [('protein', x[2]) for x in batch]
    
    list_atomresi_graphs   = [(x[3].sum(2) > 0.).astype(np.int_) for x in batch]
    list_ba_values         = [x[4] for x in batch]
    list_ba_labels         = [x[5] for x in batch]
    list_complex_indices   = [x[-1] for x in batch]

    for x in list_atomresi_graphs:
        assert len(x.shape) == 2, x.shape

    x, m, _ = stack_and_pad(list_resiwise_features)
    tensor_list.append(torch.cuda.FloatTensor(x))
    tensor_list.append(torch.cuda.FloatTensor(m))
    tensor_list.append(list_fastas)

    x, _, m = stack_and_pad_with(list_ecfp_words, padding_idx=1024) # unsafe
    tensor_list.append(torch.cuda.LongTensor(x).squeeze(1))
    tensor_list.append(torch.cuda.FloatTensor(m).squeeze(1))
    
    x, _, m = stack_and_pad(list_atomresi_graphs)
    tensor_list.append(torch.cuda.FloatTensor(x))
    tensor_list.append(torch.cuda.FloatTensor(m))

    tensor_list.append(torch.cuda.FloatTensor(list_ba_values).view(-1, 1))
    tensor_list.append(torch.cuda.FloatTensor(list_ba_labels).view(-1, 1))
    tensor_list.append(list_complex_indices)

    return tensor_list


# Needs to be fixed
class DtiDatasetInfer(DtiDatasetCore):
    def __init__(self, args):
        super().__init__(args)

    def get_test_instance(self, complex_idx):
        ligand_idx = self.complex_dataframe.loc[complex_idx, 'ligand_id']
        protein_idx = self.complex_dataframe.loc[complex_idx, 'protein_id']
        ba_value = self.complex_dataframe.loc[complex_idx, 'ba_value']
        ba_label = self.complex_dataframe.loc[complex_idx, 'ba_label']
        self.check_protein(protein_idx)

        # Ligand / Atom / Features
        mol_path = f'{self.data_path}ligands/{ligand_idx}/{ligand_idx}.mol'
        smiles = self.ligand_dataframe.loc[ligand_idx, 'smiles']
        if check_exists(mol_path): mol = pickle.load(open(mol_path, 'rb'))
        else: mol = Chem.MolFromSmiles(smiles)
        try: Chem.SanitizeMol(mol)
        except: pass 
        try:
            ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,
                                                                  nBits=self.ecfpvec_dim,
                                                                  useChirality=False))
        except:
            ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,
                                                                  nBits=self.ecfpvec_dim,
                                                                  useChirality=False))
        ecfp_words = np.nonzero(ecfp)[0].reshape(1,-1)  

        # Protein / Residue / Features
        resiwise_features = []
        fasta = self.protein_dataframe.loc[protein_idx, 'fasta']
        for resi in fasta:
            try: resiwise_features.append(self.resi_features(resi).reshape(1, -1))
            except: resiwise_features.append(np.zeros(21).reshape(1, -1))
        resiwise_features = np.vstack(resiwise_features).astype(int)    
        atomresi_graph = np.zeros((len(mol.GetAtoms()), len(fasta), 1))

        pytrdata = (ecfp_words, resiwise_features, fasta, atomresi_graph, ba_value, ba_label, complex_idx)

        return collate_fn([pytrdata])

    def get_test_instances(self, complex_idx_list):
        data_instances, meta_instances = [], []
        for complex_idx in complex_idx_list:
            try:
                ligand_idx = self.complex_dataframe.loc[complex_idx, 'ligand_id']
                protein_idx = self.complex_dataframe.loc[complex_idx, 'protein_id']
                ba_value = self.complex_dataframe.loc[complex_idx, 'ba_value']
                ba_label = self.complex_dataframe.loc[complex_idx, 'ba_label']
                self.check_protein(protein_idx)

                # Ligand / Atom / Features
                atomwise_features = []
                smiles = self.ligand_dataframe.loc[ligand_idx, 'smiles']
                try:
                    mol_path = f'{self.data_path}ligands/{ligand_idx}/{ligand_idx}.mol'
                    mol = pickle.load(open(mol_path, 'rb'))
                except:
                    mol = Chem.MolFromSmiles(smiles)
                try: Chem.SanitizeMol(mol)
                except: pass
                try:
                    ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,
                                                                          nBits=self.ecfpvec_dim,
                                                                          useChirality=False))
                except:
                    ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,
                                                                          nBits=self.ecfpvec_dim,
                                                                          useChirality=False))
                ecfp_words = np.nonzero(ecfp)[0].reshape(1,-1)
                
                # Protein / Residue / Features
                resiwise_features = []
                fasta = self.protein_dataframe.loc[protein_idx, 'fasta']
                for resi in fasta:
                    try: resiwise_features.append(self.resi_features(resi).reshape(1, -1))
                    except: resiwise_features.append(np.zeros(21).reshape(1, -1))
                resiwise_features = np.vstack(resiwise_features).astype(int)   
                atomresi_graph = np.zeros((len(mol.GetAtoms()), len(fasta), 1))

                atomresi_graph = np.zeros((atomwise_features.shape[0], len(fasta), 1))
                metadata = (complex_idx, ligand_idx, protein_idx, smiles, fasta, ba_value, ba_label)
                pytrdata = (ecfp_words, resiwise_features, fasta, atomresi_graph, ba_value, ba_label, complex_idx)

                data_instances.append(pytrdata)
                meta_instances.append(metadata)

            except Exception as e:
                pass 

        return collate_fn(data_instances), meta_instances


if __name__ == '__main__':
    print("Dataset and Dataloader for ArkDTA")