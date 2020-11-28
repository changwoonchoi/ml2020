import glob
import random

all_audios = glob.glob('../data/audio/all/*.wav')
dataset_size = len(all_audios)
test_train_ratio = 0.1

test_idx = sorted(random.sample(range(dataset_size), dataset_size // 10))

train_set = []
test_set = []

test_set_cnt = 0
for i in range(len(all_audios)):
    if  test_set_cnt < dataset_size // 10 and i == test_idx[test_set_cnt]:
        test_set.append(all_audios[i])
        test_set_cnt += 1
    else:
        train_set.append(all_audios[i])

with open('../data/train.txt', 'w') as file:
    for train_file in train_set:
        file.write(train_file + '\n')

with open('../data/test.txt', 'w') as file:
    for test_file in test_set:
        file.write(test_file + '\n')
