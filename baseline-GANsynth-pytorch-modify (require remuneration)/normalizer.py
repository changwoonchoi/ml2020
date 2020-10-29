import numpy as np
import torch 



class DataNormalizer(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader

        self._range_normalizer(magnitude_margin=0.8, IF_margin=1.0)
        print("s_a:", self.s_a )
        print("s_b:", self.s_b )
        print("p_a:", self.p_a)
        print("p_b:", self.p_b)


    def _range_normalizer(self, magnitude_margin, IF_margin):

        min_spec = 10000
        max_spec = -10000
        min_IF = 10000
        max_IF = -10000

        for batch_idx, (spec, IF, pitch_label, mel_spec, mel_IF) in enumerate(self.dataloader.train_loader): 
            
            # training mel
            spec = mel_spec
            IF = mel_IF

            
            if spec.min() < min_spec: min_spec=spec.min()
            if spec.max() > max_spec: max_spec=spec.max()

            if IF.min() < min_IF: min_IF=IF.min()
            if IF.max() > max_IF: max_IF=IF.max()

        self.s_a = magnitude_margin * (2.0 / (max_spec - min_spec))
        self.s_b = magnitude_margin * (-2.0 * min_spec / (max_spec - min_spec) - 1.0)
        
        self.p_a = IF_margin * (2.0 / (max_IF - min_IF))
        self.p_b = IF_margin * (-2.0 * min_IF / (max_IF - min_IF) - 1.0)


    def normalize(self, feature_map):
        a = np.asarray([self.s_a, self.p_a])[None, :, None, None]
        b = np.asarray([self.s_b, self.p_b])[None, :, None, None]
        a = torch.FloatTensor(a).cuda()
        b = torch.FloatTensor(b).cuda()
        feature_map = feature_map *a + b

        return feature_map

    def denormalize(spec, IF, s_a, s_b, p_a, p_b):
        spec = (spec -s_b) / s_a
        IF = (IF-p_b) / p_a
        return spec, IF