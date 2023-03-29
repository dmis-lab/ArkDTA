import torch
from torch import nn
torch.autograd.set_detect_anomaly(True)

from .base                  import *
from .residue_encoders      import *
from .ligelem_encoders      import *
from .residue_addons        import *
from .complex_decoders      import *
from .set_transformer       import * 
from .downstream_predictors import *

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.layer = nn.ModuleDict()
        analysis_mode = args.analysis_mode
        h             = args.arkdta_hidden_dim
        d             = args.hp_dropout_rate
        esm           = args.arkdta_esm_model
        esm_freeze    = args.arkdta_esm_freeze
        E             = args.arkdta_ecfpvec_dim
        L             = args.arkdta_sab_depth
        A             = args.arkdta_attention_option
        K             = args.arkdta_num_heads
        assert 'ARKMAB' in args.arkdta_residue_addon 

        self.layer['prot_encoder'] = FastaESM(h, esm, esm_freeze, analysis_mode)
        self.layer['comp_encoder'] = EcfpConverter(h, d, L, K, A, E, analysis_mode)
        self.layer['intg_arkmab']  = load_residue_addon(args)
        self.layer['intg_pooling'] = load_complex_decoder(args)
        self.layer['ba_predictor'] = AffinityMLP(h, d)
        self.layer['dt_predictor'] = InteractionMLP(h, d)

    def set_default_hp(self, trainer):

        return trainer

    def load_auxiliary_materials(self, **kwargs):
        return_batch = kwargs['return_batch']

        b = kwargs['atomresi_adj'].size(0)
        x, y, z = kwargs['encoder_attention'].size()
        logits0 = kwargs['encoder_attention'].view(x//b,b,y,z).mean(0)[:,:,:-1].sum(2).unsqueeze(2) # actual compsub
        logits1 = kwargs['encoder_attention'].view(x//b,b,y,z).mean(0)[:,:,-1].unsqueeze(2)         # inactive site
        return_batch['task/es_pred'] = torch.cat([logits1,logits0],2)
        return_batch['task/es_true'] = (kwargs['atomresi_adj'].sum(1) > 0.).long().squeeze(1)
        return_batch['mask/es_resi']  = (kwargs['atomresi_masks'].sum(1) > 0.).float().squeeze(1)

        return return_batch

    def forward(self, batch):
        return_batch = dict()
        residue_features, residue_masks, residue_fastas = batch[0], batch[1], batch[2]
        ecfp_words, ecfp_masks                          = batch[3], batch[4]
        atomresi_adj, atomresi_masks                    = batch[5], batch[6]
        bav, dti, cids                                  = batch[7], batch[8], batch[-1]

        # Protein Encoder Module
        residue_features = self.layer['prot_encoder'](X=residue_features,
                                                      fastas=residue_fastas,
                                                      masks=residue_masks)
        residue_masks    = residue_features[1]
        residue_temps    = residue_features[2]
        protein_features = residue_features[3]
        residue_features = residue_features[0]
        return_batch['temp/lm_related'] = residue_temps * 0.

        # Ligand Encoder Module
        cstruct_features = self.layer['comp_encoder'](ecfp_words=ecfp_words,
                                                      ecfp_masks=ecfp_masks)
        cstruct_masks = cstruct_features[1]
        cstruct_features = cstruct_features[0]

        # Protein-Ligand Integration Module (ARK-MAB)
        residue_results = self.layer['intg_arkmab'](residue_features=residue_features, residue_masks=residue_masks,
                                                    ligelem_features=cstruct_features, ligelem_masks=cstruct_masks)
        residue_features, residue_masks, attention_weights = residue_results
        del residue_results; torch.cuda.empty_cache()

        # Protein-Ligand Integration Module (Pooling Layer)
        complex_results = self.layer['intg_pooling'](residue_features=residue_features,
                                                     residue_masks=residue_masks,
                                                     attention_weights=attention_weights,
                                                     protein_features=protein_features)
        binding_complex, _, _, _ = complex_results
        del complex_results; torch.cuda.empty_cache()

        # Drug-Target Outcome Predictor
        bav_predicted = self.layer['ba_predictor'](binding_complex=binding_complex)
        dti_predicted = self.layer['dt_predictor'](binding_complex=binding_complex)

        return_batch['task/ba_pred'] = bav_predicted.view(-1)
        return_batch['task/dt_pred'] = dti_predicted.view(-1)
        return_batch['task/ba_true'] = bav.view(-1)
        return_batch['task/dt_true'] = dti.view(-1)
        return_batch['meta/cid']     = cids

        # Additional Materials for Calculating Auxiliary Loss
        return_batch = self.load_auxiliary_materials(return_batch=return_batch,
                                                     atomresi_adj=atomresi_adj,
                                                     atomresi_masks=atomresi_masks,
                                                     encoder_attention=attention_weights)

        return return_batch

    @torch.no_grad()
    def infer(self, batch):
        return_batch = dict()
        residue_features, residue_masks, residue_fastas = batch[0], batch[1], batch[2]
        ecfp_words, ecfp_masks                          = batch[3], batch[4]
        bav, dti, cids                                  = batch[7], batch[8], batch[-1]

        # Protein Encoder Module
        residue_features = self.layer['prot_encoder'](X=residue_features,
                                                      fastas=residue_fastas,
                                                      masks=residue_masks)
        residue_masks    = residue_features[1]
        residue_temps    = residue_features[2]
        protein_features = residue_features[3]
        residue_features = residue_features[0]
        return_batch['temp/lm_related'] = residue_temps * 0.

        # Ligand Encoder Module
        cstruct_features = self.layer['comp_encoder'](ecfp_words=ecfp_words,
                                                      ecfp_masks=ecfp_masks)
        cstruct_masks = cstruct_features[1]
        cstruct_features = cstruct_features[0]

        # Protein-Ligand Integration Module (ARK-MAB)
        residue_results = self.layer['intg_arkmab'](residue_features=residue_features, residue_masks=residue_masks,
                                                    ligelem_features=cstruct_features, ligelem_masks=cstruct_masks)
        residue_features, residue_masks, attention_weights = residue_results
        del residue_results; torch.cuda.empty_cache()

        # Protein-Ligand Integration Module (Pooling Layer)
        complex_results = self.layer['intg_pooling'](residue_features=residue_features,
                                                     residue_masks=residue_masks,
                                                     attention_weights=attention_weights,
                                                     protein_features=protein_features)
        binding_complex, _, _, _ = complex_results
        del complex_results; torch.cuda.empty_cache()

        # Drug-Target Outcome Predictor
        bav_predicted = self.layer['ba_predictor'](binding_complex=binding_complex)
        dti_predicted = self.layer['dt_predictor'](binding_complex=binding_complex)

        return_batch['task/ba_pred'] = bav_predicted.view(-1)
        return_batch['task/dt_pred'] = dti_predicted.view(-1)
        return_batch['task/ba_true'] = bav.view(-1)
        return_batch['task/dt_true'] = dti.view(-1)
        return_batch['meta/cid']     = cids

        return return_batch


if __name__ == '__main__':
    print("Model Architecture for ArkDTA")