import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("")

from train_container import *
from models import *
from utils import *
from criterion import *
from config import *

def main(**kwargs):
    data_train, data_val, data_test = load_data(kwargs['data_dir'], kwargs['augmentation'])
    data_train_shuffle, data_val_shuffle = shuffle_data(data_train, data_val)
    if kwargs['mode'] == 'pretrain':
        pipeline = [Preprocess4Mask(**kwargs)]
        model = BGBertModel4Pretrain(**kwargs)
        dataset_train = Dataset4Pretrain(data_train_shuffle, pipeline=pipeline)
        dataset_val = Dataset4Pretrain(data_val_shuffle, pipeline=pipeline)
        dataset_test = Dataset4Pretrain(data_test, pipeline=pipeline)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=kwargs['lr'])
    elif kwargs['mode'] == 'output_emb':
        pipeline = [Preprocess4Prediction(**kwargs)]
        model = BGBertModel4Pretrain(**kwargs)
        dataset_train = Dataset4Prediction(data_train_shuffle, pipeline=pipeline)
        dataset_val = Dataset4Prediction(data_val_shuffle, pipeline=pipeline)
        dataset_test = Dataset4Prediction(data_test, pipeline=pipeline)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=kwargs['lr'])
    elif kwargs['mode'] == 'prediction':
        dataset_train, dataset_val, dataset_test = load_representation(kwargs['prediction_data_dir'])
        print("dataset_train shape is", dataset_train.shape)
        model = BGBertPrediction(**kwargs)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=kwargs['lr'])

    data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=kwargs['batch_size'])
    data_loader_val = DataLoader(dataset_val, shuffle=False, batch_size=kwargs['batch_size'])
    data_loader_test = DataLoader(dataset_test, shuffle=False, batch_size=kwargs['batch_size'])

    device = get_device(kwargs['device'])
    trainer = Trainer(model, optimizer, kwargs['out_dir'], **kwargs)

    if kwargs['mode'] == 'pretrain':
        trainer.pretrain(data_loader_train, data_loader_val, model_file=None,
                         data_parallel=False, log_writer=writer)
    elif kwargs['mode'] == 'output_emb':
        train_rep, train_orig = trainer.out_emb(func_out_emb, data_loader=data_loader_train, model_file=kwargs['model_path'],
                                            data_parallel=False)
        val_rep, val_orig = trainer.out_emb(func_out_emb, data_loader=data_loader_val, model_file=kwargs['model_path'],
                                        data_parallel=False)
        test_rep, test_orig = trainer.out_emb(func_out_emb, data_loader=data_loader_test, model_file=kwargs['model_path'],
                                          data_parallel=False)
        print("train_rep shape is", train_rep.shape)
        print("train_orig shape is", train_orig.shape)
        try:
            os.makedirs(kwargs['prediction_data_dir'])
        except OSError:
            pass
        np.save(osp.join(kwargs['prediction_data_dir'], 'train_rep.npy'), train_rep.cpu().numpy()) 
        np.save(osp.join(kwargs['prediction_data_dir'], 'train_orig.npy'), train_orig.cpu().numpy())
        np.save(osp.join(kwargs['prediction_data_dir'], 'val_rep.npy'), val_rep.cpu().numpy())
        np.save(osp.join(kwargs['prediction_data_dir'], 'val_orig.npy'), val_orig.cpu().numpy())
        np.save(osp.join(kwargs['prediction_data_dir'], 'test_rep.npy'), test_rep.cpu().numpy())
        np.save(osp.join(kwargs['prediction_data_dir'], 'test_orig.npy'), test_orig.cpu().numpy())
    elif kwargs['mode'] == 'prediction':
        trainer.prediction(data_loader_train, data_loader_test, data_loader_val, model_file=None,
                           data_parallel=False)

if __name__ == "__main__":
    kwargs = load_config()
    main(**kwargs)
