import numpy as np
from pyemd import emd

def distance(std1, std2):
    '''
    std1 : distribution of std(S) for audio1
    std2 : distribution of std(S) for audio2
    output : EMD(earth mover's distance) of two distribution
    '''
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

def classify(audio):
    '''
    returns the class of stereo audio
    {NNN: 0, NNW: 1, NWN: 2, ..., WWW: 7}
    '''
    raise NotImplementedError

