import copy
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from criterion import *

class Trainer(object):
    """Training Helper Class"""
    def __init__(self, model, optimizer, save_path, **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = kwargs['device']

    def pretrain(self, data_loader_train, data_loader_test, model_file=None,
                 data_parallel=False, log_writer=SummaryWriter()):
        """ Train Loop """
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        global_step = 0
        best_loss = 1e6
        model_best = model.state_dict()
        unimproved = 0
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience = 10, factor = 0.75, verbose = True, eps = 1e-12)
        for e in range(self.kwargs['epochs']):
            loss_sum = 0.
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(tqdm(data_loader_train)):
                batch = [t.to(self.device) for t in batch]
                start_time = time.time()
                self.optimizer.zero_grad()
                loss = func_loss(model, batch, **self.kwargs)

                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                time_sum += time.time() - start_time
                global_step += 1
                loss_sum += loss.item()

                if self.kwargs['total_steps'] and self.kwargs['total_steps'] < global_step:
                    print('The Total Steps have been reached.')
                    return

            loss_eva = self.run(func_forward, func_evaluate, data_loader_test)
            scheduler.step(loss_eva)
            log_train_loss = loss_sum / len(data_loader_train)
            print('Epoch %d/%d : Average Loss %5.4f. Test Loss %5.4f'
                    % (e + 1, self.kwargs['epochs'], loss_sum / len(data_loader_train), loss_eva))
            if loss_eva < best_loss:
                unimproved = 0
                best_loss = loss_eva
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)
            else:
                unimproved += 1
                print('Unimproved: %d' % unimproved)
                if unimproved >= 20:
                    break
        model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')

    def run(self, func_forward, func_evaluate, data_loader, model_file=None, data_parallel=False, load_self=False):
        """ Evaluation Loop """
        self.model.eval()
        self.load(model_file, load_self=load_self)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        results = []
        original_datas = []
        logits_list = []
        labels_list = []
        time_sum = 0.0
        for batch in data_loader:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():
                start_time = time.time()
                result, original_data = func_forward(model, batch)
                time_sum += time.time() - start_time
                results.append(result)
                original_datas.append(original_data)
        if func_evaluate:
            return func_evaluate(torch.cat(results), torch.cat(original_datas), **self.kwargs)

        return torch.cat(results), torch.cat(original_datas)

    def out_emb(self, func_out_emb, data_loader, model_file=None, data_parallel=False, load_self=False):
        """ Evaluation Loop """
        self.model.eval()
        self.load(model_file, load_self=load_self)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        results = []
        original_datas = []
        time_sum = 0.0
        for batch in data_loader:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():
                start_time = time.time()
                result, original_data = func_out_emb(model, batch)
                time_sum += time.time() - start_time
                results.append(result)
                original_datas.append(original_data)

        return torch.cat(results), torch.cat(original_datas)

    def pred_val(self, data_loader_val, model_file=None, data_parallel=False,
                      load_self=False):
        """ Evaluation Loop """
        self.model.eval()
        self.load(model_file, load_self=load_self)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        val_loss_sum = []
        for batch in data_loader_val:
            with torch.no_grad():
                val_loss = pred_func_loss(model, batch, 'val', **self.kwargs)
                val_loss_sum.append(val_loss.mean().item())

        return np.mean(np.array(val_loss_sum))

    def prediction(self, data_loader_train, data_loader_test, data_loader_vali,
                   model_file=None, data_parallel=False, load_self=False):
        """ Train Loop """
        self.load(model_file, load_self)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience = 10, factor = 0.75, verbose = True, eps = 1e-12)
        global_step = 0 # global iteration steps regardless of epochs
        vali_loss_best = 0.0
        best_stat = None
        model_best = model.state_dict()
        unimproved = 0
        for e in range(self.kwargs['epochs']):
            loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(tqdm(data_loader_train)):

                start_time = time.time()
                self.optimizer.zero_grad()
                loss = pred_func_loss(model, batch, 'train', **self.kwargs)

                loss = loss.mean()# mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                time_sum += time.time() - start_time
                if self.kwargs['total_steps'] and self.kwargs['total_steps'] < global_step:
                    print('The Total Steps have been reached.')
                    return
            vali_loss = self.pred_val(data_loader_vali, model_file=None, data_parallel=False, load_self=False)
            scheduler.step(vali_loss)
            print('Epoch %d/%d : Train Average Loss %7.6f, Val Average Loss %7.6f'
                  % (e+1, self.kwargs['epochs'], loss_sum / len(data_loader_train), vali_loss))
            # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))
            if e == 0:
                vali_loss_best = vali_loss
            else:
                if vali_loss < vali_loss_best:
                    unimproved = 0
                    vali_loss_best = vali_loss
                    model_best = copy.deepcopy(model.state_dict())
                    self.save_pred()
                else:
                    unimproved += 1
                    print('Unimproved: %d' % unimproved)
                    if unimproved >= self.kwargs['pred_patience']:
                        break
        self.model.load_state_dict(model_best)
        self.evaluate(data_loader_test, self.model, self.kwargs['out_dir'])
        print('The Total Epoch have been reached.')

    def evaluate(self, data_loader_test, model, out_dir):
        """ Evaluation Loop """
        self.model.eval()
        model = self.model.to(self.device)

        with torch.no_grad():
            rmselosses=[] # root mean squared error
            tglosses = [] # time gain loss
            sen_hyper = [] # sensitivity of hyperglycemic event points
            sen_hypo = [] # sensitivity of hypoglycemic event points
            pred_g_list = [] # predicted glucose
            orig_g_list = [] # original glucose
            history_list = [] # history glucose
            for i, batch in enumerate(tqdm(data_loader_test)):
                rmse, tg, sen_h, sen_l, pred_g, orig_g = pred_func_eval(model, batch, **self.kwargs)
                rmselosses.append(rmse)
                tglosses.append(tg)
                sen_hyper.append(sen_h)
                sen_hypo.append(sen_l)
                pred_g_list.append(pred_g)
                orig_g_list.append(orig_g)
                history_list.append(batch[0].cpu().numpy())

        rmse = np.sqrt(np.mean(np.array(rmselosses)))
        tg = np.mean(np.array(tglosses))
        sen_h = np.sum(np.concatenate(sen_hyper)) / len(np.concatenate(sen_hyper))
        sen_l = np.sum(np.concatenate(sen_hypo)) / len(np.concatenate(sen_hypo))
        print('RMSE: {:.4f}, TG: {:.4f}, sen_hyper: {:.4f}, sen_hypo: {:.4f}'.format(rmse, tg, sen_h, sen_l))
        pred_g_list = np.concatenate(pred_g_list)
        orig_g_list = np.concatenate(orig_g_list)
        history_list = np.concatenate(history_list)
        np.save(out_dir + "/" + "pred-g.npy", pred_g_list)
        np.save(out_dir + "/" + "orig-g.npy", orig_g_list)

        t = open(out_dir + "/" + str(rmse) + ".Rmseout","w")
        t = open(out_dir + "/" + str(tg) + ".Tgout","w")
        t = open(out_dir + "/" + str(sen_h) + ".Senhyperout","w")
        t = open(out_dir + "/" + str(sen_l) + ".Senhypoout","w")

    def load(self, model_file, load_self=False):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            if load_self:
                self.model.load_self(model_file + '.pt', map_location=self.device)
            else:
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))

    def save(self, i=0):
        """ save current model """
        if i != 0:
            torch.save(self.model.state_dict(), self.save_path + "_" + str(i) + '.pt')
        else:
            torch.save(self.model.state_dict(),  self.save_path + '/0.pt')

    def save_pred(self):
        """ save current model """
        torch.save(self.model.state_dict(), self.save_path + '/best_pred.pt')

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
