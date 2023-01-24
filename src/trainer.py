from typing import Any, Callable, List, Tuple, Union

from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.distributed as dist
from pytorch_lamb import Lamb
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW as adamw
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from utils import *

from time import sleep
import pickle

# from env_config import *
# from data_utils import *
# from model_utils import calculate_sensitivity as calc_sens

import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score as aup
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index as c_index
import pandas as pd

torch.autograd.set_detect_anomaly(True)

SCHEDULER_FIRSTSTEP = [None]
SCHEDULER_BATCHWISE = [None]
SCHEDULER_EPOCHWISE = [None]

class Trainer:
    def __init__(self, args, rank, wandb_run=None, ddp_mode=True):
        self.args = args

        self.debug_mode = args.debug_mode
        self.ddp_mode   = ddp_mode
        self.rank = rank
        self.wandb = wandb_run 
        self.pred_model = args.pred_model

        self.num_epochs    = args.hp_num_epochs
        self.learning_rate = args.hp_learning_rate
        self.weight_decay  = args.hp_weight_decay
        self.temp_scalar   = args.hp_temp_scalar
        self.aux_coef      = args.hp_aux_coefficient
        self.main_coef     = args.hp_main_coefficient

        if args.ba_measure == 'Binary': 
            self.reg_coef, self.clf_coef = 0.0, self.main_coef
            self.main_index = 1
        else:
            self.reg_coef, self.clf_coef = self.main_coef, 0.0
            self.main_index = 0

        self.best_eval_loss = np.Inf
        self.num_patience = args.hp_early_patience

        self.checkpoint_path = f'{args.checkpoint_path}_fold_{args.fold_num}_mea_{args.ba_measure}' 
        if not self.args.testing_mode:
            os.makedirs(self.checkpoint_path, exist_ok=True)

        self.save_name = args.session_name
        self.lookup_values = dict({'true_aff': [], 'pred_aff': [], 
                                   'true_dti': [], 'pred_dti': [],
                                   'true_pwi': [], 'pred_pwi': [], 'meta_cid': []})
        self.aux_weights = [1.0, args.arkdta_posweighted]

        if args.dataset_subsets == 'bddb': self.aux_coef = 0.0
        self.dataset_subsets = args.dataset_subsets

        if self.rank == 0:
            print("Loss Coefficient for Regression Part:    ", self.reg_coef)
            print("Loss Coefficient for Classification Part:", self.clf_coef)
            print("Loss Coefficient for Auxiliary Part:     ", self.aux_coef)

    def calculate_losses(self, batch):
        total_loss = []#nn.ModuleList([])
        ba_criterion = nn.MSELoss().to(self.rank)
        dt_criterion = DDP_BCELoss(self.rank)
        es_criterion = Masked_NLLLoss(self.rank, self.aux_weights)

        if 'task/ba_true' in batch.keys():
            total_loss.append(ba_criterion(batch['task/ba_pred'], batch['task/ba_true']) * self.reg_coef)

        if 'task/dt_true' in batch.keys():
            total_loss.append(dt_criterion(batch['task/dt_pred'], batch['task/dt_true']) * self.clf_coef)

        if 'task/es_true' in batch.keys():
            total_loss.append(es_criterion(batch['task/es_pred'], batch['task/es_true'],
                                           batch['mask/es_resi']) * self.aux_coef)

        if 'temp/lm_related' in batch.keys():
            total_loss.append(batch['temp/lm_related'])

        return total_loss 

    def check_wrong_losses(self, train_loss, valid_loss):
        if not np.isfinite(train_loss): print("ABNORMAL TRAIN LOSS"); return True
        if not np.isfinite(valid_loss): print("ABNORMAL VALID LOSS"); return True
        if train_loss < 0: print("NEGATIVE TRAIN LOSS"); return True
        if valid_loss < 0: print("NEGATIVE VALID LOSS"); return True
        return False

    def reset_lookup_values(self):

        self.lookup_values = dict({'true_aff': [], 'pred_aff': [], 
                                   'true_dti': [], 'pred_dti': [],
                                   'true_pwi': [], 'pred_pwi': [], 'meta_cid': []})

    def store_lookup_values(self, batch):
        self.lookup_values['meta_cid'].extend(batch['meta/cid'])

        if 'task/ba_true' in batch.keys():
            self.lookup_values['true_aff'].extend(numpify(batch['task/ba_true']))
            self.lookup_values['pred_aff'].extend(numpify(batch['task/ba_pred']))

        if 'task/dt_true' in batch.keys():
            self.lookup_values['true_dti'].extend(numpify(batch['task/dt_true']))
            self.lookup_values['pred_dti'].extend(numpify(batch['task/dt_pred']))

        if 'task/es_true' in batch.keys():
            for i in range(batch['task/es_true'].size(0)):
                if batch['mask/es_resi'][i,:].sum() != 0:
                    es_true = batch['task/es_true'][i,:].view(-1) 
                    es_pred = batch['task/es_pred'][i,:,1].view(-1)
                    es_mask = batch['mask/es_resi'][i,:].view(-1)
                    labels = numpify(es_true[es_mask>0.])
                    logits = numpify(es_pred[es_mask>0.])

                    self.lookup_values['true_pwi'].append(labels)
                    self.lookup_values['pred_pwi'].append(logits)

    def wandb_lookup_values(self, label, wandb_dict):
        if not self.ddp_mode: return
        num_ranks = torch.cuda.device_count()

        if len(self.lookup_values['true_aff']) > 0:
            y, yhat = [None for _ in range(num_ranks)], [None for _ in range(num_ranks)]
            dist.all_gather_object(y,    self.lookup_values['true_aff'])
            dist.all_gather_object(yhat, self.lookup_values['pred_aff'])
            y, yhat = np.hstack(y), np.hstack(yhat)

            wandb_dict[f'{label}/aff/mae']  = mae(y,yhat)
            wandb_dict[f'{label}/aff/rmse'] = mse(y,yhat) ** .5
            pearson = pearsonr(y,yhat)[0]
            spearman = spearmanr(y,yhat)[0]
            if np.isnan(pearson): pearson = 0
            if np.isnan(spearman): spearman = 0  
            wandb_dict[f'{label}/aff/pearson'] = pearson
            wandb_dict[f'{label}/aff/spearman'] = spearman
            wandb_dict[f'{label}/aff/ci'] = c_index(y,yhat)

            cids = [None for _ in range(num_ranks)]
            dist.all_gather_object(cids, self.lookup_values['meta_cid'])
            cids = sum(cids, [])

            if label in ['test', 'hardt'] and self.rank == 0:
                data = np.hstack([y.reshape(-1,1),yhat.reshape(-1,1)])
                df = pd.DataFrame(data, columns=['true_affinity', 'pred_affinity'], index=cids)
                df.to_csv(self.checkpoint_path + f'/results_{self.dataset_subsets}_{label}_aff.csv')

        if len(self.lookup_values['true_dti']) > 0:
            y, yhat = [None for _ in range(num_ranks)], [None for _ in range(num_ranks)]
            dist.all_gather_object(y,    self.lookup_values['true_dti'])
            dist.all_gather_object(yhat, self.lookup_values['pred_dti'])
            y, yhat = np.hstack(y), np.hstack(yhat)

            try:    wandb_dict[f'{label}/dti/auroc'] = auc(y.astype(int),yhat)
            except: wandb_dict[f'{label}/dti/auroc'] = 0.0
            try:    wandb_dict[f'{label}/dti/auprc'] = aup(y.astype(int),yhat)
            except: wandb_dict[f'{label}/dti/auprc'] = 0.0
            wandb_dict[f'{label}/dti/f1score']   = f1(y.astype(int),(yhat>0.5).reshape(-1))
            wandb_dict[f'{label}/dti/accuracy']  = acc(y.astype(int),(yhat>0.5).reshape(-1))

            cids = [None for _ in range(num_ranks)]
            dist.all_gather_object(cids, self.lookup_values['meta_cid'])
            cids = sum(cids, [])

            if label in ['test', 'hardt'] and self.rank == 0:
                data = np.hstack([y.reshape(-1,1),yhat.reshape(-1,1)])
                df = pd.DataFrame(data, columns=['true_interaction', 'pred_interaction'], index=cids)
                df.to_csv(self.checkpoint_path + f'/results_{self.dataset_subsets}_{label}_dti.csv')

        if len(self.lookup_values['true_pwi']) > 0:
            auc_list, aup_list, f1_list, acc_list = [], [], [], []
            for pw_true, pw_pred in zip(self.lookup_values['true_pwi'], self.lookup_values['pred_pwi']):
                auc_list.append(auc(pw_true, pw_pred))
                aup_list.append(aup(pw_true, pw_pred))
                pw_pred = (pw_pred > 0.5).reshape(-1) 
                f1_list.append(f1(pw_true, pw_pred))
                acc_list.append(acc(pw_true, pw_pred))

            auc_gathered = [None for _ in range(num_ranks)]
            aup_gathered = [None for _ in range(num_ranks)]
            f1_gathered  = [None for _ in range(num_ranks)]
            acc_gathered  = [None for _ in range(num_ranks)]
            dist.all_gather_object(auc_gathered, auc_list)
            dist.all_gather_object(aup_gathered, aup_list)
            dist.all_gather_object(f1_gathered,  f1_list)
            dist.all_gather_object(acc_gathered, acc_list)
            auc_gathered = sum(auc_gathered, [])
            aup_gathered = sum(aup_gathered, [])
            f1_gathered  = sum(f1_gathered, [])
            acc_gathered = sum(acc_gathered, [])

            wandb_dict[f'{label}/pwi/auroc']    = np.mean(auc_gathered)
            wandb_dict[f'{label}/pwi/auprc']    = np.mean(aup_gathered)
            wandb_dict[f'{label}/pwi/f1score']  = np.mean(f1_gathered)
            wandb_dict[f'{label}/pwi/accuracy'] = np.mean(acc_gathered)

        if self.rank == 0: self.wandb.log(wandb_dict)

        return

    def get_optimizer(self, model):
        if self.args.pred_model in ['arkdta']:
            optimizer = adamw(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)  
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, 
                                                       weight_decay=self.weight_decay)

        return optimizer

    def get_scheduler(self):
        if self.args.pred_model == 'arkdta':
            scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.train_steps, 
                                                        num_training_steps=self.train_steps*self.num_epochs)
        else: scheduler = DummyScheduler()
        return scheduler

    def train_valid(self, model, train, train_sampler=None, valid=None, hard_valid=None):
        self.train_steps = len(train)
        num_ranks = torch.cuda.device_count()
        print(f"RANK: {self.rank+1} | Training Batches: {len(train)}, Validation Batches: {len(valid)}")
        EARLY_STOPPING = False

        model = model.to(self.rank)
        if not self.debug_mode: model = DDP(model, device_ids=[self.rank])

        self.optimizer = self.get_optimizer(model)
        self.scheduler = self.get_scheduler()

        for epoch in range(self.num_epochs):
            if train_sampler: train_sampler.set_epoch(epoch)
            train_loss, model = self.train_step(model, train, epoch)
            self.wandb_lookup_values('train', {'train/loss': train_loss.item(), 'train/step': epoch}) 
            self.reset_lookup_values()

            eval_loss, _ = self.eval_step(model, valid)
            self.wandb_lookup_values('valid', {'valid/loss': eval_loss.item(), 'valid/step': epoch})
            self.reset_lookup_values()

            hard_loss, _ = self.eval_step(model, hard_valid)
            self.wandb_lookup_values('hardv', {'hardv/loss': hard_loss.item(), 'hardv/step': epoch})
            self.reset_lookup_values()

            train_loss_gathered = [None for _ in range(num_ranks)]
            valid_loss_gathered = [None for _ in range(num_ranks)]
            hardv_loss_gathered = [None for _ in range(num_ranks)] 
            dist.all_gather_object(train_loss_gathered, train_loss)
            dist.all_gather_object(valid_loss_gathered, eval_loss )
            dist.all_gather_object(hardv_loss_gathered, hard_loss )

            if self.rank == 0:
                train_loss = np.mean(train_loss_gathered)
                eval_loss  = np.mean(valid_loss_gathered)
                hard_loss  = np.mean(hardv_loss_gathered)
                print(f"Epoch: {epoch+1}, Train: {np.mean(train_loss_gathered):.3f}, Valid: {np.mean(valid_loss_gathered):.3f}, Hard Valid: {np.mean(hardv_loss_gathered):.3f}")

            if self.check_wrong_losses(train_loss, eval_loss):
                EARLY_STOPPING = True
            if eval_loss > self.best_eval_loss:
                if self.num_patience > 0: self.num_patience -= 1
                else: EARLY_STOPPING = True
            else:
                self.best_eval_loss = eval_loss
                if self.rank == 0:
                    if not self.debug_mode:
                        torch.save(model.module.state_dict(), self.checkpoint_path + f'/best_epoch.mdl')
            if EARLY_STOPPING: break

        if self.rank == 0: 
            if not self.debug_mode:
                torch.save(model.module.state_dict(), self.checkpoint_path + f'/last_epoch.mdl')
            self.wandb.finish()

        return model, train_loss, eval_loss

    def train_step(self, model, data, epoch=0):
        model.train()
        if not self.args.debug_mode:
            torch.distributed.barrier()
        batchwise_loss = [] 

        if self.pred_model in SCHEDULER_FIRSTSTEP:
            self.scheduler.step()
        for idx, batch in enumerate(data):
            self.optimizer.zero_grad()
            batch = model(batch)

            loss = self.calculate_losses(batch)
            sum(loss).backward() 
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            if self.pred_model in SCHEDULER_BATCHWISE: 
                self.scheduler.step()

            sleep(0.01)
            batchwise_loss.append(loss[self.main_index].item())
            self.store_lookup_values(batch)
            del batch; torch.cuda.empty_cache()
        if self.pred_model in SCHEDULER_EPOCHWISE:
            self.scheduler.step()

        return np.mean(batchwise_loss), model

    @torch.no_grad()
    def eval_step(self, model, data):
        model.eval()
        batchwise_loss = []

        for idx, batch in enumerate(data):
            batch = model(batch)            
            loss = self.calculate_losses(batch)

            sleep(0.01)
            batchwise_loss.append(loss[self.main_index].item())
            self.store_lookup_values(batch)
            del batch; torch.cuda.empty_cache()

        return np.mean(batchwise_loss), model

    @torch.no_grad()
    def test(self, model, test, hard_test):
        print(f"RANK: {self.rank} | Test Batches: {len(test)}")
        model = model.to(self.rank)
        model = DDP(model, device_ids=[self.rank])
        print("Testing Model on RANK: ", self.rank)
        eval_loss, _ = self.eval_step(model, test)
        self.wandb_lookup_values('test', {'test/loss': eval_loss.item()})
        self.reset_lookup_values()

        print("Hard-Testing Model on RANK: ", self.rank)
        eval_loss, _ = self.eval_step(model, hard_test)
        self.wandb_lookup_values('hardt', {'hardt/loss': eval_loss.item()})
        self.reset_lookup_values()
        if self.rank == 0: self.wandb.finish()

        return eval_loss.item()

class ArkTrainer(Trainer):
    def __init__(self, args, rank, wandb_run=None, ddp_mode=True):
        super(ArkTrainer, self).__init__(args, rank, wandb_run, ddp_mode)

    def train_step(self, model, data, epoch):
        model.train()
        if not self.args.debug_mode:
            torch.distributed.barrier()
        batchwise_loss = [] 

        for idx, batch in enumerate(data):
            self.optimizer.zero_grad()
            batch = model(batch)

            loss = self.calculate_losses(batch)
            sum(loss).backward() 
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.scheduler.step()

            sleep(0.01)
            batchwise_loss.append(loss[self.main_index].item())
            self.store_lookup_values(batch)
            del batch; torch.cuda.empty_cache()


        return np.mean(batchwise_loss), model