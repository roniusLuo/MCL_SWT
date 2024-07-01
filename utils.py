"""
Utilities in Mirror contrastive loss based sliding window transformer.

Written by Jing Luo from Xi'an University of Technology, China.

luojing@xaut.edu.cn
"""
import numpy as np
import os
import os.path
import torch
import torch.nn as nn
from collections import OrderedDict
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from bcic_iv_2b import BCICompetition4Set2B
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.signal_target import SignalAndTarget
import math
import warnings


def read_data_2a(data_folder, low_cut_hz, high_cut_hz):
    train_set, test_set = process_2a(data_folder, 1, low_cut_hz, high_cut_hz)
    for subject_id in range(2, 10, 1):
        temp_train_set, temp_test_set = process_2a(data_folder, subject_id, low_cut_hz, high_cut_hz)
        train_set = merge_set(temp_train_set, train_set)
        test_set = merge_set(temp_test_set, test_set)
    train_set.X = np.expand_dims(train_set.X, 1)
    test_set.X = np.expand_dims(test_set.X, 1)
    return train_set, test_set


def read_data_2b(data_folder, low_cut_hz, high_cut_hz):
    test_set = process_2b(data_folder, 1, low_cut_hz, high_cut_hz)
    for subject_id in range(2, 10, 1):
        temp_set = process_2b(data_folder, subject_id, low_cut_hz, high_cut_hz)
        test_set = merge_set(test_set, temp_set)
    test_set.X = np.expand_dims(test_set.X, 1)
    return test_set


def getMirrorEEG(EEG, label):
    mirror_eeg = torch.zeros(EEG.shape, device=EEG.device, dtype=torch.float32)
    mirror_label = torch.zeros(EEG.shape[0], device=EEG.device, dtype=torch.float32)
    mirror_eeg[:, :, (0, 1, 2), :] = EEG[:, :, (2, 1, 0), :]
    where = torch.eq(label, 0)
    mirror_label[where] = 1
    return mirror_eeg, mirror_label


def merge_set(sat1, sat2):
    tempX = np.concatenate((sat1.X, sat2.X))
    tempY = np.concatenate((sat1.y, sat2.y))
    return SignalAndTarget(tempX, tempY)


def pick2class(data_set):
    tempX = data_set.X
    tempY = data_set.y
    trialNum = tempY.size
    for t in range(trialNum - 1, -1, -1):
        if tempY[t] in [2, 3]:
            tempX = np.delete(tempX, t, axis=0)
            tempY = np.delete(tempY, t, axis=0)
    return SignalAndTarget(tempX, tempY)


def process_2a(data_folder, subject_id, low_cut_hz, high_cut_hz):
    ival = [-500, 3980]
    factor_new = 1e-3
    init_block_size = 1000
    # Data loading
    train_filename = 'A{:02d}T.gdf'.format(subject_id)
    test_filename = 'A{:02d}E.gdf'.format(subject_id)
    train_filepath = os.path.join(data_folder, train_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    train_label_filepath = train_filepath.replace('.gdf', '.mat')
    test_label_filepath = test_filepath.replace('.gdf', '.mat')
    train_loader = BCICompetition4Set2A(
        train_filepath, labels_filename=train_label_filepath)
    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath)
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()

    # train_cnt
    train_cnt = train_cnt.pick_channels(['EEG-C3', 'EEG-C4', 'EEG-Cz'])
    assert len(train_cnt.ch_names) == 3
    # convert to millvolt for numerical stability of next operations
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    # bandpass
    train_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        train_cnt)

    # test_cnt
    test_cnt = test_cnt.pick_channels(['EEG-C3', 'EEG-C4', 'EEG-Cz'])
    assert len(test_cnt.ch_names) == 3
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    test_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt)
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        test_cnt)
    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])
    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
    # left and right 2-category EEG
    train_set = pick2class(train_set)

    test_set = pick2class(test_set)
    return train_set, test_set


def process_2b(data_folder, subject_id, low_cut_hz, high_cut_hz):
    ival = [-500, 3980]
    factor_new = 1e-3
    init_block_size = 1000

    test_filename4 = 'B{:02d}04E.gdf'.format(subject_id)
    test_filename5 = 'B{:02d}05E.gdf'.format(subject_id)
    test_filepath4 = os.path.join(data_folder, test_filename4)
    test_filepath5 = os.path.join(data_folder, test_filename5)
    test_label_filepath4 = test_filepath4.replace('.gdf', '.mat')
    test_label_filepath5 = test_filepath5.replace('.gdf', '.mat')
    test_loader4 = BCICompetition4Set2B(
        test_filepath4, labels_filename=test_label_filepath4)
    test_loader5 = BCICompetition4Set2B(
        test_filepath5, labels_filename=test_label_filepath5)
    test_cnt4 = test_loader4.load()

    test_cnt5 = test_loader5.load()
    # Preprocessing
    test_cnt4 = test_cnt4.pick_channels(['EEG:C3', 'EEG:C4', 'EEG:Cz'])
    assert len(test_cnt4.ch_names) == 3
    test_cnt4 = mne_apply(lambda a: a * 1e6, test_cnt4)
    test_cnt4 = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt4.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt4)
    test_cnt4 = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        test_cnt4)

    test_cnt5 = test_cnt5.pick_channels(['EEG:C3', 'EEG:C4', 'EEG:Cz'])
    assert len(test_cnt5.ch_names) == 3
    test_cnt5 = mne_apply(lambda a: a * 1e6, test_cnt5)
    test_cnt5 = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt5.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt5)
    test_cnt5 = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        test_cnt5)
    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])

    tempSet4 = create_signal_target_from_raw_mne(test_cnt4, marker_def, ival)
    tempSet5 = create_signal_target_from_raw_mne(test_cnt5, marker_def, ival)
    test_set = merge_set(tempSet4, tempSet5)
    return test_set


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-linear_len, 2u-linear_len].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class MCL(nn.Module):
    def __init__(self, alpha=0.2, beta=1.2):
        super(MCL, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, featureO, featureM, labelO, labelM):
        distance_matrix = torch.cdist(featureO, featureM, p=2)
        g = torch.zeros((labelO.size(0), labelM.size(0)), device='cuda')
        for i in range(labelO.size(0)):
            for j in range(labelM.size(0)):
                if labelO[i] == labelM[j]:
                    g[i][j] = 1
                else:
                    g[i][j] = -1

        loss_partial = torch.clamp(self.alpha + g * (distance_matrix - self.beta), min=0)
        loss = torch.sum(loss_partial) / (labelO.size(0) * labelM.size(0))
        return loss
