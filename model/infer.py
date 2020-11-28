from PGGAN import *
import torch.optim as optim
import torch.backends.cudnn as cudnn
import librosa.display

import torch.utils.data as udata
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import spec_ops as spec_ops
import phase_operation as phase_op
import spectrograms_helper as spec_helper
from normalizer import DataNormalizer

import scipy.io.wavfile as wv
import numpy as np
import argparse

def polar2rect(mag, phase_angle):
    """Convert polar-form complex number to its rectangular form."""
    temp_mag = np.zeros(mag.shape,dtype=np.complex_)
    temp_phase = np.zeros(mag.shape,dtype=np.complex_)

    for i, time in enumerate(mag):
        for j, time_id in enumerate(time):
            temp_mag[i,j] = np.complex(mag[i,j])

    for i, time in enumerate(phase_angle):
        for j, time_id in enumerate(time):
            temp_phase[i,j] = np.complex(np.cos(phase_angle[i,j]), np.sin(phase_angle[i,j]))

    return temp_mag * temp_phase

def mag_plus_phase(mag, IF):

    mag =  np.exp(mag) - 1.0e-6
    reconstruct_magnitude = np.abs(mag)
    reconstruct_phase_angle = np.cumsum(IF * np.pi, axis=1)
    stft = polar2rect(reconstruct_magnitude, reconstruct_phase_angle)
    inverse = librosa.istft(stft, hop_length = 512, win_length=2048, window = 'hann')

    return inverse

def denormalize(spec, IF, s_a, s_b, p_a, p_b):
    spec = (spec -s_b) / s_a
    IF = (IF-p_b) / p_a
    return spec, IF

def output_file(model,seed, pitch):
    fake_pitch_label = torch.LongTensor(1, 1).random_() % 8
    pitch = [[pitch]]
    fake_pitch_label = torch.LongTensor(pitch)
    fake_one_hot_pitch_condition_vector = torch.zeros(1, 8).scatter_(1, fake_pitch_label, 1).unsqueeze(2).unsqueeze(3).cuda()
    fake_pitch_label = fake_pitch_label.cuda().squeeze()

    fake_seed_and_pitch_condition = torch.cat((seed, fake_one_hot_pitch_condition_vector), dim=1)
    output = model(fake_seed_and_pitch_condition)
    output = output.squeeze()
    spec_L = output[0].data.cpu().numpy().T
    IF_L = output[1].data.cpu().numpy().T
    spec_L, IF_L = denormalize(spec_L, IF_L, s_a=0.060437, s_b=0.034964, p_a=0.0034997, p_b=-0.010897)
    back_mag_L, back_IF_L = spec_helper.melspecgrams_to_specgrams(spec_L, IF_L)
    back_mag_L = np.vstack((back_mag_L,back_mag_L[1023]))
    back_IF_L = np.vstack((back_IF_L,back_IF_L[1023]))
    audio_L = mag_plus_phase(back_mag_L,back_IF_L)

    spec_R = output[2].data.cpu().numpy().T
    IF_R = output[3].data.cpu().numpy().T
    spec_R, IF_R = denormalize(spec_R, IF_R, s_a=0.060437, s_b=0.034964, p_a=0.0034997, p_b=-0.010897)
    back_mag_R, back_IF_R = spec_helper.melspecgrams_to_specgrams(spec_R, IF_R)
    back_mag_R = np.vstack((back_mag_R,back_mag_R[1023]))
    back_IF_R = np.vstack((back_IF_R,back_IF_R[1023]))
    audio_R = mag_plus_phase(back_mag_R,back_IF_R)
    return audio_L, audio_R

def gen_audio(gnet, seed, pitch, type):
    ad_L, ad_R = output_file(gnet, seed, pitch)
    if type == 'MS':
        ad_M = np.array(ad_L)
        ad_S = np.array(ad_R)
        ad_L = ad_M + ad_S
        ad_R = ad_M - ad_S
    elif type == 'LR':
        ad_L = np.array(ad_L)
        ad_R = np.array(ad_R)
    else:
        raise ValueError
    ad_L *= 32767
    ad_R *= 32767
    ad_L_int16 = ad_L.astype(np.int16)
    ad_R_int16 = ad_R.astype(np.int16)
    audioL = []
    audioR = []
    for left in ad_L_int16:
        adl = left
        if adl > 32767:
            adl = 32767
        elif adl < -32768:
            adl = -32768
        audioL.append(adl)
    for right in ad_R_int16:
        adr = right
        if adr > 32767:
            adr = 32767
        elif adr < -32768:
            adr = -32768
        audioR.append(adr)

    audio = [audioL, audioR]
    audio = np.transpose(audio)
    return audio


parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--type')
parser.add_argument('--sample_num')
parser.add_argument('--save_dir')
args = parser.parse_args()

g_net = Generator(256, 256, 4, is_tanh=True,channel_list=[256,256,256,256,256,128,64,32])
g_checkpoint = torch.load(args.model)
g_net.load_state_dict(g_checkpoint)
g_net.net_config = [6, 'stable', 1]
g_net.cuda()

for i in range(int(args.sample_num)):
    fake_seed = torch.randn(1, 256, 1, 1).cuda()
    audio = gen_audio(g_net, fake_seed, 3, args.type)
    wv.write(os.path.join(args.save_dir, '{}.wav'.format(i)),44100, audio)
