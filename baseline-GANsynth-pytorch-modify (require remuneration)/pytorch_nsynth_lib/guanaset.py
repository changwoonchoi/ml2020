import os
# import csv
import glob
import numpy as np
import scipy.io.wavfile
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

pitch_table = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

class Gset(data.Dataset):

    def __init__(self, root, transform=None):
        """Constructor"""
        assert(isinstance(root, str))
        self.root = root
        self.filenames = glob.glob(os.path.join(root, "audio/*.wav"))
        """Labeling 참조하는거 때려치고 걍 뽑자"""
#         with open(os.path.join(root, "pitchList.csv"), newline='') as csvfile:
#             self.csvdata = csv.DictReader(self.csvfile)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        _, sample = scipy.io.wavfile.read(name)
        sample_left = []
        sample_right = []
        for i in range(0, len(sample)):
            sample_left.append(sample[i][0])
            sample_right.append(sample[i][1])
        sample_left = np.array(sample_left)
        sample_right = np.array(sample_right)

#         sample_left.astype(np.int16)
#         sample_right.astype(np.int16)
        
        if self.transform is not None:
            sample_left = self.transform(sample_left)
            sample_right = self.transform(sample_right)
        
#         for row in csvdata:
#             if name == os.path.join(self.root, "audio", row['file_name']):
#                 pitch_data = row['pitch']

        """pitch 뽑은 뒤의 comma는 dtype때문에 뜨더라"""
        
        pitch_data_char = os.path.splitext(name)[0].split('_')[1]
        pitch_data = pitch_table.index(pitch_data_char) + 24
        
        return sample_left, sample_right, pitch_data


if __name__ == "__main__":
    # audio samples are loaded as an int16 numpy array
    # rescale intensity range as float [-1, 1]
    toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)
    # use instrument_family and instrument_source as classification targets
    dataset = Gset(
        "../GUANADATA",
        transform=toFloat)
    loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
#     for samples, instrument_family_target, instrument_source_target, targets \
#             in loader:
#         print(samples.shape, instrument_family_target.shape,
#               instrument_source_target.shape)
#         print(torch.min(samples), torch.max(samples))
