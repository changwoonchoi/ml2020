import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from pytorch_nsynth_lib.guanaset import Gset
from IPython.display import Audio
import os
import librosa
import librosa.display
import phase_operation
import spectrograms_helper as spec_helper
import matplotlib.pyplot as plt

toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)
dataset = Gset("./", transform=toFloat)
loader = data.DataLoader(dataset, batch_size=1, shuffle=True)

def expand(mat):
    expand_vec = np.expand_dims(mat[:,34],axis=1)
    expanded = np.hstack((mat,expand_vec))
    return expanded

def reduce(mat):
    return np.delete(mat, np.s_[32:], axis=1)

spec_list=[]
pitch_list=[]
IF_list =[]
mel_spec_list=[]
mel_IF_list=[]

pitch_set =set()
count=0
for samples_left, samples_right, pitches in loader:

    sample_left = samples_left.data.numpy().squeeze()
    sample_right = samples_right.data.numpy().squeeze()

    sample_mid = (sample_left + sample_right) / 2
    sample_side = (sample_left - sample_right) / 2

    sample_left = sample_mid
    sample_right = sample_side
    spec_left = librosa.stft(sample_left, n_fft=2048, hop_length = 512)
    spec_right = librosa.stft(sample_right, n_fft=2048, hop_length = 512)
#     print(spec_left[100])

    """ log10 말고 log로 처리"""
    magnitude_left = np.log(np.abs(spec_left)+ 1.0e-6)[:1024]
    magnitude_right = np.log(np.abs(spec_right)+ 1.0e-6)[:1024]

    angle_left = np.angle(spec_left)
    angle_right = np.angle(spec_right)

    IF_left = phase_operation.instantaneous_frequency(angle_left,time_axis=1)[:1024]
    IF_right = phase_operation.instantaneous_frequency(angle_right,time_axis=1)[:1024]

    magnitude_left = reduce(magnitude_left)
    magnitude_right = reduce(magnitude_right)

    IF_left = reduce(IF_left)
    IF_right = reduce(IF_right)

    logmelmag2_left, mel_p_left = spec_helper.specgrams_to_melspecgrams(magnitude_left, IF_left)
    logmelmag2_right, mel_p_right = spec_helper.specgrams_to_melspecgrams(magnitude_right, IF_right)

#     print(magnitude_left.shape)

    assert magnitude_left.shape ==(1024, 32)
    assert magnitude_right.shape ==(1024, 32)
    assert IF_left.shape ==(1024, 32)
    assert IF_right.shape ==(1024, 32)

    pitch_list.append(pitches)
    mel_spec_list.append([logmelmag2_left,logmelmag2_right])
    mel_IF_list.append([mel_p_left,mel_p_right])
    pitch_set.add(pitches)

    count+=1
    if count%1000==0:
        print(count)
    if count%4000==0:
        np.save('../data/np_array_train_MS/' + 'pitch_list_' + str(count), pitch_list)
        np.save('../data/np_array_train_MS/' + 'mel_spec_list_' + str(count), mel_spec_list)
        np.save('../data/np_array_train_MS/' + 'mel_IF_list_' + str(count), mel_IF_list)
        pitch_list=[]
        mel_spec_list=[]
        mel_IF_list=[]
        print(str(count) + " saved")
np.save('../data/np_array_train_MS/' + 'pitch_list_' + str(count), pitch_list)
np.save('../data/np_array_train_MS/' + 'mel_spec_list_' + str(count), mel_spec_list)
np.save('../data/np_array_train_MS/' + 'mel_IF_list_' + str(count), mel_IF_list)
print(str(count) + " saved")
