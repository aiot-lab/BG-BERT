import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import copy
from torch.utils.data import Dataset, TensorDataset, DataLoader

def mse_loss(pred, target):
    criterion = nn.MSELoss(reduction='none')

    return criterion(pred, target)

def focal_loss(pred, target):
    criterion = FocalLoss()

    return criterion(pred, target)

def compute_labels(mask_seqs, masked_pos, seqs):
    """
    Three levels:
        hypo: 0, (glucose level < 70)
        normal: 1, (70 <= glucose level <= 180)
        hyper: 2 (glucose level > 180)
    """
    labels = torch.zeros((len(mask_seqs), 3))
    # put seqs back to mask_seqs using masked_pos
    for i in range(len(masked_pos)):
        mask_seqs[:, masked_pos[i], :] = seqs[i]
    # compute the labels
    for i in range(len(mask_seqs)):
        for j in range(mask_seqs.shape[1]):
            if torch.sum(mask_seqs[i, j, :] < 70):
                labels[i, 0] += 1

    return labels

def func_loss(model, batch, **kwargs):
    # criterion = ShrinkageLoss(**kwargs)
    mask_seqs, masked_pos, seqs = batch #
    seq_recon = model(mask_seqs, masked_pos) #
    loss_lm = mse_loss(seq_recon, seqs) # for masked LM
    # loss_lm = criterion(seq_recon, seqs) # for masked LM

    return loss_lm

def func_forward(model, batch):
    mask_seqs, masked_pos, seqs= batch
    seq_recon = model(mask_seqs, masked_pos)

    return seq_recon, seqs

def func_out_emb(model, batch):
    mask_seqs, masked_pos, seqs = batch
    seq_recon = model(mask_seqs, masked_pos)

    return seq_recon, seqs

def func_evaluate(seqs, predict_seqs, **kwargs):
    criterion = nn.MSELoss(reduction='none')
    # criterion = ShrinkageLoss(**kwargs)
    loss_lm = criterion(predict_seqs, seqs)
    print("loss_lm is: ", loss_lm.mean())

    return loss_lm.mean().cpu().numpy()

def rmse_loss(pred, target):
    """
    pred: (batch_size, horizon)
    target: (batch_size, horizon)
    """
    criterion = nn.MSELoss(reduction='none')

    return torch.sqrt(criterion(pred, target))

def mse_cpu(output, target):
    return np.mean((output - target)**2)

def tg_cpu(output, target, horizon):
    tg = []
    for i in range(len(target)):
        min_error = np.inf
        best_k = 0
        for k in range(horizon):
            error = np.sum((output[i, k:] - target[i, :horizon-k])**2) / (horizon - k)
            if error < min_error:
                min_error = error
                best_k = k
        tg.append(horizon - best_k)

    return np.mean(np.array(tg)) * 5

def event_hyper(output, target):
    # calculate how many hyper events there are
    eventpoints = (target > 250)
    # check whether the prediction results is also hyper
    pred_results = output[eventpoints]
    # if the prediction is also hyper, then set it to 1
    # otherwise, set it to 0
    pred_results = (pred_results > 250).astype(int)

    return pred_results

def event_hypo(output, target):
    # calculate how many hypo events there are
    eventpoints = (target < 70)
    # check whether the prediction results is also hypo
    pred_results = output[eventpoints]
    # if the prediction is also hypo, then set it to 1
    # otherwise, set it to 0
    pred_results = (pred_results < 70).astype(int)

    return pred_results

def pred_func_loss(model, batch, pred_mode, **kwargs):
    # criterion = ShrinkageLoss(**kwargs)
    # criterion = mse_loss()
    criterion = CovWeighting(**kwargs)
    # seperate the features and original glucose levels
    # print("batch size is: ", batch.shape)
    features = batch[:, :, :-kwargs['feature_num']]
    original_glucose = batch[:, 0:kwargs['fore_length'], -kwargs['feature_num']]
    original_glucose = original_glucose.squeeze(-1)
    # put into tensor
    features = features.to(kwargs['device'])
    original_glucose = original_glucose.to(kwargs['device'])
    # get the predicted glucose levels
    pred_glucose = model(features)

    # compute the loss
    if pred_mode == 'train':
        loss = criterion.get_loss(pred_glucose, original_glucose, 'train')
    elif pred_mode == 'val':
        loss = criterion.get_loss(pred_glucose, original_glucose, 'val')
    # loss = criterion(pred_glucose, original_glucose)
    # loss = mse_loss(pred_glucose, original_glucose)

    return loss

def pred_func_eval(model, batch, **kwargs):
    # seperate the features and original glucose levels
    features = batch[:, :, :-kwargs['feature_num']]
    original_glucose = batch[:, 0:kwargs['fore_length'], -kwargs['feature_num']]
    original_glucose = original_glucose.squeeze(-1)
    features = features.to(kwargs['device'])
    original_glucose = original_glucose.to(kwargs['device'])
    # get the predicted glucose levels
    pred_glucose = model(features)
    # regenerate glucose level
    pred_glucose = pred_glucose * 400
    original_glucose = original_glucose * 400
    # compute the loss
    rmse = mse_cpu(pred_glucose.cpu().numpy(), original_glucose.cpu().numpy())
    tg = tg_cpu(pred_glucose.cpu().numpy(), original_glucose.cpu().numpy(), kwargs['fore_length'])
    sen_h = event_hyper(pred_glucose.cpu().numpy(), original_glucose.cpu().numpy())
    sen_l = event_hypo(pred_glucose.cpu().numpy(), original_glucose.cpu().numpy())

    return rmse, tg, sen_h, sen_l, pred_glucose.cpu().numpy(), original_glucose.cpu().numpy()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLoss()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = logits
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class ShrinkageLoss(nn.Module):
    """
    Shrinkage loss for glucose level prediction.
    L = l * l / (1 + exp(a * (c - l)))
    l: absolute error
    a: shrinkage parameter
    """
    def __init__(self, **kwargs):
        super(ShrinkageLoss, self).__init__()
        self.device = kwargs.get('device')
        self.kwargs = kwargs

    def forward(self, predicts, targets):
        """
        Params:
            predicts: glucose levels prediction, [batchsize, horizon]
            targets: ground truth glucose levels, [batchsize, horizon]
        """
        # shrinkage parameter
        a = 0.2
        c = 1
        # absolute error
        if self.kwargs['mode'] == 'pretrain':
            l = torch.abs(predicts - targets)
        elif self.kwargs['mode'] == 'prediction':
            l = torch.abs(predicts - targets) / targets
        # l = torch.abs(predicts - targets)
        # print(l.shape)
        # print(torch.exp(targets).shape)
        # shrinkage loss
        loss = l * l / (1 + torch.exp(a * (c - l)))

        return loss.mean()

class CosSim(nn.Module):
    """
    cos similarity loss
    """
    def __init__(self, **kwargs):
        super(CosSim, self).__init__()
        self.device = kwargs.get('device')
    
    def forward(self, predicts, targets):
        """
        Params:
            predicts: glucose levels prediction, [batchsize, horizon]
            targets: ground truth glucose levels, [batchsize, horizon]
        """
        # computer each result in the formula: (a * b) / (|a| * |b|)
        a = predicts
        b = targets
        loss = 1 - (torch.sum(a * b, dim=1) / (torch.sqrt(torch.sum(a * a, dim=1)) * torch.sqrt(torch.sum(b * b, dim=1))))

        return loss.mean()

class CoVWeightingLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CoVWeightingLoss, self).__init__()
        self.device = kwargs.get('device')
        self.num_losses = 2
        self.shrinkage = ShrinkageLoss(**kwargs)
        self.diff_loss = nn.MSELoss(reduction='none')

    def forward(self, predicts, targets):
        """
        Params:
            predicts: velocities of shape (batch_size, length, 2)
            targets: velocities of shape (batch_size, length, 2)
        """
        # rmse loss
        shrink_loss = self.shrinkage(predicts, targets)
        # diff loss
        diff_loss = self.diff_loss(torch.diff(predicts, dim=1), torch.diff(targets, dim=1))
        diff_loss = diff_loss.mean()
        # print("shrink_loss is: ", shrink_loss)
        # print("diff_loss is: ", diff_loss)
        loss = [shrink_loss, diff_loss]

        return loss

class CovWeighting():
    def __init__(self, **kwargs):
        super(CovWeighting, self).__init__()
        self.device = kwargs.get('device')
        # How to compute the mean statistics: Full mean or decaying mean.
        self.mean_decay = False
        self.mean_decay_param = 1.0

        self.criterion = CoVWeightingLoss(**kwargs)
        self.current_iter = -1
        self.num_losses = 2
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_std_l = None

    def get_loss(self, predicts, targets, option):
        # Retrieve the unweighted losses.
        unweighted_losses = self.criterion.forward(predicts, targets)
        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if option == 'val':
            return torch.sum(L) / self.num_losses

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device) / self.num_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        # elif self.current_iter > 0 and self.mean_decay:
        #     mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        loss = sum(weighted_losses)

        return loss

    def get_val_loss(self, predicts, targets):
        # Retrieve the unweighted losses.
        unweighted_losses = self.criterion.forward(predicts, targets)
        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        return torch.sum(L)
