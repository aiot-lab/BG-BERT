import os
import random
from typing import Any
import torch
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def logging(file):
    def write_log(s):
        print(s)
        with open(file, 'a') as f:
            f.write(s+'\n')
    return write_log

def set_up_logging(**kwargs):
    if kwargs['out_dir'] is None:
        raise ValueError('out_dir is needed')
    if kwargs['model_type'] is None:
        raise ValueError('model_type is needed')
    if kwargs['dataset'] is None:
        raise ValueError('dataset is needed')
    log = logging(os.path.join(kwargs["out_dir"], kwargs["model_type"] + "_" + kwargs["dataset"] + '.txt'))

    log("%s:\t%s\n" % (str(kwargs["model_type"]), str(kwargs["dataset"])))

    return log

def format_string(*argv, sep=' '):
    result = ''
    for val in argv:
        if isinstance(val, (tuple, list, np.ndarray)):
            for v in val:
                result += format_string(v, sep=sep) + sep
        else:
            result += str(val) + sep

    return result[:-1]

def get_device(gpu):
    "get device (CPU or GPU)"
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))

    return device

def compute_labels(dataset):
    """
    Three levels:
        hypo: 0, (glucose level < 70)
        normal: 1, (70 <= glucose level <= 250)
        hyper: 2 (glucose level > 250)
    """
    print("dataset shape is", dataset.shape)
    labels = np.zeros((len(dataset)))
    # compute the labels
    for i in range(len(dataset)):
        if np.sum(dataset[i, :, 0] < 70):
            labels[i] = 0
        elif np.sum(dataset[i, :, 0] > 250):
            labels[i] = 2
        else:
            labels[i] = 1

    return labels

def data_augmentation(data):
    """
    Augment training data via SMOTE
    """
    labels = compute_labels(data)
    counter = Counter(labels)
    print("before augmentation, labels are", counter)
    # over = SMOTE(sampling_strategy={0:40000, 2:50000}) # For OhioT1DM
    over = SMOTE(sampling_strategy={0:50000, 2:90000}) # For Diatrend
    steps = [('o', over)]
    pipeline = Pipeline(steps=steps)
    total_num = data.shape[0]
    seq_len = data.shape[1]
    num_feature = data.shape[2]
    data = np.array(data).reshape(total_num, seq_len * num_feature)
    data_aug, labels_aug = pipeline.fit_resample(data, labels)
    counter = Counter(labels_aug)
    print("after augmentation, labels are", counter)
    data_aug = data_aug.reshape(data_aug.shape[0], seq_len, num_feature)

    return data_aug

def load_data(data_dir, augmentation):
    train_data = np.load(osp.join(data_dir, 'train.npy'))
    val_data = np.load(osp.join(data_dir, 'val.npy'))
    test_data = np.load(osp.join(data_dir, 'test.npy'))

    combined_data = np.concatenate((train_data, val_data, test_data), axis=0)
    max_cgm = np.max(combined_data[:, :, 0])
    print("max_cgm is", max_cgm)
    max_1 = np.max(combined_data[:, :, 1])
    max_meal = np.max(combined_data[:, :, 2])
    max_insulincarbratio = np.max(combined_data[:, :, 3])
    max_4 = np.max(combined_data[:, :, 4])

    if augmentation:
        train_data = data_augmentation(train_data)
    train_data[:, :, 0] = train_data[:, :, 0] / max_cgm
    train_data[:, :, 1] = train_data[:, :, 1] / max_1
    train_data[:, :, 2] = train_data[:, :, 2] / max_meal
    train_data[:, :, 3] = train_data[:, :, 3] / max_insulincarbratio
    train_data[:, :, 4] = train_data[:, :, 4] / max_4

    val_data[:, :, 0] = val_data[:, :, 0] / max_cgm
    val_data[:, :, 1] = val_data[:, :, 1] / max_1
    val_data[:, :, 2] = val_data[:, :, 2] / max_meal
    val_data[:, :, 3] = val_data[:, :, 3] / max_insulincarbratio
    val_data[:, :, 4] = val_data[:, :, 4] / max_4

    test_data[:, :, 0] = test_data[:, :, 0] / max_cgm
    test_data[:, :, 1] = test_data[:, :, 1] / max_1
    test_data[:, :, 2] = test_data[:, :, 2] / max_meal
    test_data[:, :, 3] = test_data[:, :, 3] / max_insulincarbratio
    test_data[:, :, 4] = test_data[:, :, 4] / max_4

    return train_data, val_data, test_data

def load_representation(data_dir):
    train_rep = np.load(osp.join(data_dir, 'train_rep.npy'))
    train_orig = np.load(osp.join(data_dir, 'train_orig.npy'))
    total_len = train_rep.shape[1]
    horizon = train_orig.shape[1]
    train_orig = np.concatenate([train_orig] * (total_len // horizon), axis=1)
    dataset_train = np.concatenate((train_rep, train_orig), axis=2)
    val_rep = np.load(osp.join(data_dir, 'val_rep.npy'))
    val_orig = np.load(osp.join(data_dir, 'val_orig.npy'))
    val_orig = np.concatenate([val_orig] * (total_len // horizon), axis=1)
    dataset_val = np.concatenate((val_rep, val_orig), axis=2)
    test_rep = np.load(osp.join(data_dir, 'test_rep.npy'))
    test_orig = np.load(osp.join(data_dir, 'test_orig.npy'))
    test_orig = np.concatenate([test_orig] * (total_len // horizon), axis=1)
    dataset_test = np.concatenate((test_rep, test_orig), axis=2)

    return dataset_train, dataset_val, dataset_test

def shuffle_data(data_train, data_val):
    data_train_shuffle = data_train.copy()
    data_val_shuffle = data_val.copy()

    np.random.shuffle(data_train_shuffle)
    np.random.shuffle(data_val_shuffle)

    return data_train_shuffle, data_val_shuffle

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def bert_mask(seq_len, goal_num_predict):
    return random.sample(range(seq_len), goal_num_predict)

def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)

class Preprocess4Mask:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, **kwargs):
        self.mask_ratio = kwargs['mask_ratio']  # masking probability
        self.mask_alpha = kwargs['mask_alpha']
        self.max_gram = kwargs['mask_gram']
        self.mask_prob = kwargs['mask_prob']
        self.replace_prob = kwargs['replace_prob']

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data
    
    def compute_labels(self, mask_seqs, masked_pos, seqs):
        """
        Three levels:
            hypo: 0, (glucose level < 70)
            normal: 1, (70 <= glucose level <= 180)
            hyper: 2 (glucose level > 180)
        """
        labels = torch.zeros((3))
        for i in range(len(masked_pos)):
            mask_seqs[masked_pos[i], :] = seqs[i]
        if np.sum((mask_seqs[:, 0] * 400) < 70) :
            labels[0] += 1
        elif np.sum((mask_seqs[:, 0] * 400) > 180):
            labels[2] += 1
        else:
            labels[1] += 1
        # print(labels)
        return labels

    def __call__(self, instance):
        shape = instance.shape
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))
        mask_pos = span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)
        instance_mask = instance.copy()

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            if np.random.rand() < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[1])
            elif np.random.rand() < self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[1])
        else:
            mask_pos_index = mask_pos
            if np.random.rand() < self.mask_prob:
                instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
            elif np.random.rand() < self.replace_prob:
                instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        seq = instance[mask_pos_index, :]

        return instance_mask, np.array(mask_pos_index), np.array(seq)

class Preprocess4Prediction:
    def __init__(self, **kwargs):
        self.back_length = kwargs['back_length']
        self.fore_length = kwargs['fore_length']

    def __call__(self, instance):
        mask_pos_index = np.arange(self.back_length, self.back_length + self.fore_length)
        instance_mask = instance.copy()
        instance_mask[mask_pos_index, :] = np.zeros((len(mask_pos_index), instance.shape[1]))
        seq = instance[mask_pos_index, :]

        return instance_mask, np.array(mask_pos_index), np.array(seq)

class Dataset4Pretrain(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        mask_seq, masked_pos, seq = instance

        return torch.from_numpy(mask_seq).float(), torch.from_numpy(masked_pos).long(), torch.from_numpy(seq).float()

    def __len__(self):
        return len(self.data)

class Dataset4Prediction(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        mask_seq, masked_pos, seq = instance

        return torch.from_numpy(mask_seq).float(), torch.from_numpy(masked_pos).long(), torch.from_numpy(seq).float()

    def __len__(self):
        return len(self.data)
