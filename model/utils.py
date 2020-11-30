import os
from glob import glob
import numpy as np
import torch
from pyemd import emd
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io.wavfile
from scipy.stats import wasserstein_distance


def make_set(dir, type='dir'):
    """
    type : 'dir', 'txt'
    returns stacked npy file that contains whole wav file in directory
    """
    if type == 'dir':
        wavfiles = glob(os.path.join(dir, '*.wav'))
    elif type == 'txt':
        wavfiles = []
        txt_file = open(dir, 'r')
        while True:
            line = txt_file.readline()[:-1]
            if not line:
                break
            wavfiles.append(line)
        txt_file.close()
    else:
        raise ValueError
    wave_set = []
    for wave_instance in wavfiles:
        _, sample = scipy.io.wavfile.read(wave_instance)
        audio = sample.T / 2**15
        wave_set.append(audio)
    wave_set = np.stack(wave_set, axis=0)
    return wave_set


def side_std(audio):
    """
    audio : np array of size (N x 2)
    output : normalized distribution of Side sound for input audio
    """
    t_index, stds = std_over_time(audio, sr=44100, hop_size=1, threshold=0.125, display=False, continue_plot=False)
    return stds


def normalize_std(std):
    cum = 0
    for i in range(std.shape[0]):
        cum += std[i]
    std *= (10000. / cum)
    return std


def distance(std1, std2):
    """
    std1 : distribution of std(S) for audio1
    std2 : distribution of std(S) for audio2
    output : EMD(earth mover's distance) of two distribution
    """
    # normalize std1, std2 to convert into probabilistic distribution function
    std1_norm = normalize_std(std1)
    std2_norm = normalize_std(std2)
    # calculate EMD between two pdfs
    d = wasserstein_distance(std1_norm, std2_norm)
    return d


def dist_matrix(set1, set2):
    """
    set_1 : normal set of inferred audios
    set_2 : normal set of reference audios (have same size with set_1)
    output :  distance matrix between two audio sets
    """
    g = set1.shape[0]
    r = set2.shape[0]
    gen_ref_set = torch.cat((set1, set2), dim=0).view((g + r, 1, set1.shape[1], -1))
    dist_mat = torch.zeros((g + r, g + r))
    for i in range(g + r):
        for j in range(i + 1, g + r):
            dist_mat[i][j] = distance(gen_ref_set[i], gen_ref_set[j])

    return g, r, dist_mat


def cov(dist_mat):
    """
    set_1 : normal set of inferred audios
    set_2 : normal set of reference audios (have same size with set_1)
    output : calculated Coverage between inferred set and reference set
    """
    raise NotImplementedError


def mmd(dist_mat):
    """
    set_1 : normal set of inferred audios
    set_2 : normal set of reference audios (have same size with set_1)
    output : calculated MMD(Minimum Matching Distance) between inferred set and reference set
    """
    raise NotImplementedError


def k_nna(k, dist_mat):
    """
    k : parameter for k-NNA(k-nearest neighbor accuracy)
    set_1 : normal set of inferred audios
    set_2 : normal set of reference audios (have same size with set_1)
    output : calculated k-NNA
    """
    raise NotImplementedError


def jsd(set_1, set_2):
    """
    set_1 : normal set of inferred audios
    set_2 : normal set of reference audios (have same size with set_1)
    output : JSD(Jensen-Shannon Divergence) between two sets
    """


def classify(audio):
    """
    returns the class of stereo audio
    {NNN: 0, NNW: 1, NWN: 2, ..., WWW: 7}
    """
    raise NotImplementedError


def std_over_time(audio, sr=44100, win_size=1764, hop_size=None,
                  threshold=0.125, display=False, continue_plot=False):
    """
    ## inputs
    audio: np array of size 2 x 44100*0.4
    win_size: (default)40ms
    hop_size: (default)0.25*win_size
    threshold: (~50% percentile of N-W criteria)0.125
    display: True for show figure
    continue_plot: True for many plots in one figure

    ## outputs
    t_index, stds
    """
    if hop_size == None:
        hop_size = int(win_size / 4)
    theta = np.pi / 4
    trans = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
    points = np.dot(trans, audio)[0]
    stds = []
    t_index = np.arange(int(win_size / 2), int(len(points) - win_size / 2), hop_size) / sr
    for t in np.arange(0, int(len(points) - win_size), hop_size):
        stds.append(np.std(points[t:t + win_size]))
    stds = np.array(stds)
    if display:
        if not continue_plot:
            plt.figure(figsize=(15, 8))
        plt.plot(t_index, stds, linewidth=1)
        plt.axhline(threshold, color='r')
        plt.xlim([0, 0.4])
        plt.ylim([0, 0.5])
        plt.title('std moving window')
        plt.xlabel('time(s)')
        plt.ylabel('standard deviation')
        if not continue_plot:
            plt.show()
    return t_index, stds


def find_label(file_path):
    '''
    file_path : single .wav file path or file name as string
    output : label('N' or 'W') of .wav(ex: ['N','W','W'])
    '''
    label = pd.read_csv('../data/label.csv', index_col=0)
    file_name = file_path.split('/')[-1]
    return label.loc[file_name, :].values
