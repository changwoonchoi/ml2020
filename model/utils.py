import numpy as np
from pyemd import emd
import matplotlib.pyplot as plt

def distance(std1, std2):
    '''
    std1 : distribution of std(S) for audio1
    std2 : distribution of std(S) for audio2
    output : EMD(earth mover's distance) of two distribution
    '''
    # normalize std1, std2 to convert into probabilistic distribution function

    # calculate EMD between two pdfs

    raise NotImplementedError

def side_std(audio):
    '''
    audio : np array of size (N x 2)
    output : normalized distribution of Side sound for input audio
    '''
    raise NotImplementedError

def cov(set_1, set_2):
    '''
    set_1 : normal set of inferred audios
    set_2 : normal set of reference audios (have same size with set_1)
    output : calculated Coverage between inferred set and reference set
    '''
    raise NotImplementedError

def mmd(set_1, set_2):
    '''
    set_1 : normal set of inferred audios
    set_2 : normal set of reference audios (have same size with set_1)
    output : calculated MMD(Minimum Matching Distance) between inferred set and reference set
    '''
    raise NotImplementedError

def k_NNA(k, set_1, set_2):
    '''
    k : parameter for k-NNA(k-nearest neighbor accuracy)
    set_1 : normal set of inferred audios
    set_2 : normal set of reference audios (have same size with set_1)
    output : calculated k-NNA
    '''
    raise NotImplementedError

def jsd(set_1, set_2):
    '''
    set_1 : normal set of inferred audios
    set_2 : normal set of reference audios (have same size with set_1)
    output : JSD(Jensen-Shannon Divergence) between two sets
    '''

def classify(audio):
    '''
    returns the class of stereo audio
    {NNN: 0, NNW: 1, NWN: 2, ..., WWW: 7}
    '''
    raise NotImplementedError


### function std_over_time

def std_over_time(audio, sr=44100, win_size = 1764, hop_size = None,
                  threshold = 0.125, display=False, continue_plot=False):
    '''
    ## inputs
    audio: np array of size 2 x 44100*0.4
    win_size: (default)40ms
    hop_size: (default)0.25*win_size
    threshold: (~50% percentile of N-W criteria)0.125
    display: True for show figure
    continue_plot: True for many plots in one figure

    ## outputs
    t_index, stds
    '''
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
    if display == True:
        if continue_plot == False:
            plt.figure(figsize=(15, 8))
        plt.plot(t_index, stds, linewidth=1)
        plt.axhline(threshold, color='r')
        plt.xlim([0, 0.4])
        plt.ylim([0, 0.5])
        plt.title('std moving window')
        plt.xlabel('time(s)')
        plt.ylabel('standard deviation')
        if continue_plot == False:
            plt.show()
    return t_index, stds
