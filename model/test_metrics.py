from utils import *
import scipy.io.wavfile
import numpy as np
import glob
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gen_dir')

args = parser.parse_args()

# read reference set (test set)
ref_set_list = []
ref_set_list_file = open('../data/test.txt', 'r')
while True:
    line = ref_set_list_file.readline()[:-1]
    if not line:
        break
    ref_set_list.append(line)
ref_set_list_file.close()

ref_set = []
for ref_set_instance in ref_set_list:
    _, sample = scipy.io.wavfile.read(ref_set_instance)
    audio = sample.T
    ref_set.append(audio)
ref_set = np.stack(ref_set, axis=0)

# read generated set (inferred audios)
gen_set_file_list = glob.glob(os.path.join(args.gen_dir, '*.wav'))
gen_set = []
for gen_set_instance in gen_set_file_list:
    _, sample = scipy.io.wavfile.read(gen_set_instance)
    audio = sample.T
    gen_set.append(audio)
gen_set = np.stack(gen_set, axis=0)

# test std
stds_1 = side_std(gen_set[0])
stds_2 = side_std(gen_set[1])
d = distance(stds_1, stds_2)

# test distance
ref_set = make_set('../data/test.txt', type='txt')
gen_set = make_set(args.gen_dir, type='dir')
ref_set = ref_set[:100, :, :15872]
ref_std_set = wave_set2std_set(ref_set)
gen_std_set = wave_set2std_set(gen_set)
dist_mat = dist_matrix(gen_std_set, ref_std_set)
breakpoint()
# test MMD
MMD = mmd(dist_mat)
# test cov
COV = cov(dist_mat)
# test k-NNA
k_NNA = k_nna(dist_mat)
# test JSD
