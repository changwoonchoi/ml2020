import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io.wavfile
from scipy.stats import wasserstein_distance

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

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
        sample = cut_audio(sample)
        audio = sample.T / 2**15
        wave_set.append(audio)
    wave_set = np.stack(wave_set, axis=0)
    return wave_set


def cut_audio(audio):
    """
    cut your audio which is loaded with scipy to 15872 length
    """
    audio = audio[:15872]
    return audio


def wave_set2std_set(wave_set):
    """
    convert wave set to set of stds over time
    wave_set : (N x 2 x -1)
    output : (N x -1)
    """
    std_set = []
    for i in range(wave_set.shape[0]):
        wave = wave_set[i]
        std = side_std(wave)
        std_set.append(std)
    std_set = np.stack(std_set, axis=0)
    return std_set


def side_std(audio):
    """
    audio : np array of size (N x 2)
    output : normalized distribution of Side sound for input audio
    """
    t_index, stds = std_over_time(audio, sr=44100, hop_size=1, threshold=0.125, display=False, continue_plot=False)

    return stds


def wave_set2score_set(wave_set, metric='quantile'):
    """
    convert wave set to set of scores over time
    wave_set : (N x 2 x -1)
    metric : [‘std’, ‘quantile’, ‘pca ratio’, ‘pca angle’, ‘lr angle’, ‘correlation’]
    output : (N x -1)
    """
    score_set = []
    for i in range(wave_set.shape[0]):
        wave = wave_set[i]
        _, score = score_over_time(wave, metric=metric, sr=44100, hop_size=None, display=False, continue_plot=False)
        score_set.append(score)
    score_set = np.stack(score_set, axis=0)
    return score_set


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


def dist_matrix(gen_set, ref_set):
    """
    gen_set : normal set of inferred audios (G x 44100 * 0.4)
    ref_set : normal set of reference audios (R x 44100 * 0.4) (i.e. testset)
    output: distance matrix between two audio sets
            (G + R) x (G + R) upper triangular matrix
            dist_mat[i][j] = distance(gen+ref[i], gen+ref[j])
    """
    g = gen_set.shape[0]
    r = ref_set.shape[0]

    gen_ref_set = np.concatenate((gen_set, ref_set), axis=0).reshape(g + r, gen_set.shape[1])
    dist_mat = np.zeros((g + r, g + r))

    for i in range(g + r):
        if i%100==0:
            # prevent too frequent status printing
            print('{} / {}'.format(i, g + r))
        for j in range(i + 1, g + r):
            dist_mat[i][j] = distance(gen_ref_set[i], gen_ref_set[j])

    return g, r, dist_mat


def cov(g, r, dist_mat):
    """
    g : |G| (size of generated set)
    r : |R| (size of reference set)
    dist_mat : distance matrix
    output : calculated Coverage between generated set and reference set
    """
    matched_idx = np.zeros(g)
    for i in range(g):
        matched_idx[i] = np.argmin(dist_mat[i, g:])
    return np.unique(matched_idx).shape[0] / r


def mmd(g, r, dist_mat):
    """
    g : |G| (size of generated set)
    r : |R| (size of reference set)
    dist_mat : distance matrix
    output : calculated MMD (Minimum Matching Distance) between inferred set and reference set
    """
    mmd_val = 0
    for i in range(r):
        mmd_val += np.min(dist_mat[:g, g + i])
    return mmd_val / r


def k_nna(g, r, dist_mat, k=1):
    """
    k : parameter for k-NNA(k-nearest neighbor accuracy)
    g : |G| (size of generated set)
    r : |R| (size of reference set)
    dist_mat : distance matrix
    output : calculated k-NNA
    """
    if k == 1:
        k_nna_val = 0
        for i in range(g + r):
            min_idx = np.argmin(np.concatenate((dist_mat[:i, i], np.array([100000.]), dist_mat[i, i+1:])))
            if i < g:
                if min_idx < g:
                    k_nna_val += 1
            else:
                if min_idx >= g:
                    k_nna_val += 1
    else:
        raise NotImplementedError
    return k_nna_val / (g + r)


def jsd(set_1, set_2):
    """
    set_1 : normal set of inferred audios
    set_2 : normal set of reference audios (have same size with set_1)
    output : JSD(Jensen-Shannon Divergence) between two sets
    """
    raise NotImplementedError


def classify(audio):
    """
    returns the class of stereo audio
    {NNN: 0, NNW: 1, NWN: 2, ..., WWW: 7}
    """
    label = {'NNN': 0, 'NNW': 1, 'NWN': 2, 'NWW': 3, 'WNN': 4, 'WNW': 5,
             'WWN': 6, 'WWW': 7}
    str = ''
    step = audio.shape[1]

    raise NotImplementedError


def stereo_classification(stereo_audio, th=0.125):
    """
    stereo_audio: numpy array with size (2 X N)
    th: threshold of std
    output : array of string with length 3
    """
    time_split_mark = [0, 2940, 8820, 15872]
    trans = np.array([[np.cos(np.pi/4),-np.sin(np.pi/4)],[np.sin(np.pi/4),np.cos(np.pi/4)]])
    stereo_class = []
    for i in range(len(time_split_mark)-1):
        splited_audio = stereo_audio[:, time_split_mark[i]:time_split_mark[i+1]]
        split_std = np.std(np.dot(trans, splited_audio)[0])
        if split_std > th:
            classed = "W"
            stereo_class = np.append(stereo_class, np.array([classed]))
        else:
            classed = "N"
            stereo_class = np.append(stereo_class, np.array([classed]))
    return stereo_class


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


def score_over_time(audio, metric = 'std', sr=44100, win_size=1764, hop_size=None,
                  display=False, continue_plot=False):
    """
    ## inputs
    audio: np array of size 2 x 44100*0.4
    metric: [‘std’, ‘quantile’, ‘pca ratio’, ‘pca angle’, ‘lr angle’, ‘correlation’]
    win_size: (default)40ms
    hop_size: (default)0.25*win_size
    display: True for show figure
    continue_plot: True for many plots in one figure

    ## outputs
    t_index, scores
    """
    if hop_size == None:
        hop_size = int(win_size / 4)
    l = audio.shape[1]
    t_index = np.arange(int(win_size / 2), int(l - win_size / 2), hop_size) / sr
    scores = []
    
    if metric == 'std':
        theta = np.pi / 4
        trans = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
        points = np.dot(trans, audio)[0]
        for t in np.arange(0, int(l - win_size), hop_size):
            scores.append(np.std(points[t:t + win_size]))
            
    elif metric == 'quantile':
        theta = np.pi / 4
        trans = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
        points = np.dot(trans, audio)[0]
        for t in np.arange(0, int(l - win_size), hop_size):
            scores.append(np.diff(np.quantile(points[t:t + win_size],[0.05,0.95]))[0])
            
    elif metric == 'pca ratio':
        ratio = []        
        for t in np.arange(0, int(l - win_size), hop_size):
            L = audio[0,t:t + win_size]+1e-5*np.random.randn(win_size) # small perturbation for aviod inf
            R = audio[1,t:t + win_size]+1e-5*np.random.randn(win_size)
            pca = PCA(n_components=2)
            pca.fit(np.vstack((L,R)).T)
            if pca.components_[0,0]*pca.components_[0,1]>0:
                r = pca.explained_variance_[0]/pca.explained_variance_[1]
            else:
                r = pca.explained_variance_[1]/pca.explained_variance_[0]
    #             print('reverse')
            ratio = np.append(ratio, r)    
        scores = np.log(ratio)   
        
    elif metric == 'pca angle':
        slope = []        
        for t in np.arange(0, int(l - win_size), hop_size):
            L = audio[0,t:t + win_size]
            R = audio[1,t:t + win_size]
            pca = PCA(n_components=2)
            pca.fit(np.vstack((L,R)).T)
            a = pca.components_[0,1]/pca.components_[0,0]
            a = np.arctan(a)
            if (a >= -np.pi/2) and (a <= -np.pi/4): # set range of slope as [-90,90]
                a = a+np.pi
            a = a*180/np.pi-45 # set y=x as degree of zero
            slope = np.append(slope, a) 
        scores = slope
        
    elif metric == 'lr angle':
        slope = []
        for t in np.arange(0, int(l - win_size), hop_size):
            L = audio[0,t:t + win_size]
            R = audio[1,t:t + win_size]
            reg = LinearRegression(fit_intercept=False).fit(L.reshape(-1,1), R)
            a = reg.coef_[0]
            a = np.arctan(a)
            if (a >= -np.pi/2) and (a <= -np.pi/4): # set range of slope as [-90,90]
                a = a+np.pi
            a = a*180/np.pi-45 # set y=x as degree of zero
            slope = np.append(slope, a) 
        scores = slope

    elif metric == 'correlation':
        for t in np.arange(0, int(l - win_size), hop_size):
            L = audio[0,t:t + win_size]
            R = audio[1,t:t + win_size]
            c = pd.DataFrame(np.vstack((L,R)).T).corr().iloc[0,1]
            scores  = np.append(scores , c)
            
    if display:
        if not continue_plot:
            plt.figure(figsize=(15, 8))
        plt.plot(t_index, scores, linewidth=1)
        plt.xlim([0, 0.4])
        if metric =='std':
            plt.ylim([0, 0.5])
            plt.axhline(0.125,color='r')
        elif metric =='pca ratio':
            plt.ylim([-5, 10])
            plt.axhline(0,color='r')
        elif metric =='pca angle' or metric =='lr angle':
            plt.ylim([-90, 90])
            plt.axhline(0,color='r')
        elif metric =='quantile':
            plt.ylim([0, 1])
            plt.axhline(0.25,color='r')
        elif metric =='correlation':
            plt.ylim([-1, 1])
            plt.axhline(0.0,color='r')
        plt.title(f'{metric} moving window')
        plt.xlabel('time(s)')
        plt.ylabel(f'{metric}')
        if not continue_plot:
            plt.show()
    return t_index, scores


def find_label(file_path):
    '''
    file_path : single .wav file path or file name as string
    output : label('N' or 'W') of .wav(ex: ['N','W','W'])
    '''
    label = pd.read_csv('../data/label.csv', index_col=0)
    file_name = file_path.split('/')[-1]
    return label.loc[file_name, :].values
