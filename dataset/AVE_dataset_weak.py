import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class AVEDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(AVEDataset, self).__init__()
        self.split = split
        self.visual_feature_path = os.path.join(data_root, 'visual_feature.h5')
        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')
        self.noisy_visual_feature_path = os.path.join(data_root, 'visual_feature_noisy.h5') # only background
        self.noisy_audio_feature_path = os.path.join(data_root, 'audio_feature_noisy.h5')   # only background
        # Now for the supervised task
        self.labels_path = os.path.join(data_root, 'labels.h5') # original labels for testing
        self.dir_labels_path = os.path.join(data_root, 'mil_labels.h5')  # video-level labels
        self.dir_labels_bg_path = os.path.join(data_root, 'labels_noisy.h5')  # only background
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.h5_isOpen = False

    def __getitem__(self, index):
        if not self.h5_isOpen:
            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.visual_feature = h5py.File(self.visual_feature_path, 'r')['avadataset']
            self.audio_feature = h5py.File(self.audio_feature_path, 'r')['avadataset']
            self.clean_labels = h5py.File(self.dir_labels_path, 'r')['avadataset']

            if self.split == 'train':
                self.negative_labels = h5py.File(self.dir_labels_bg_path, 'r')['avadataset']
                self.negative_visual_feature = h5py.File(self.noisy_visual_feature_path, 'r')['avadataset']
                self.negative_audio_feature = h5py.File(self.noisy_audio_feature_path, 'r')['avadataset']

            if self.split == 'test':
                self.labels = h5py.File(self.labels_path, 'r')['avadataset']

            self.h5_isOpen = True

        clean_length = len(self.sample_order)
        if index >= clean_length:
            valid_index = index - clean_length
            visual_feat = self.negative_visual_feature[valid_index]
            audio_feat = self.negative_audio_feature[valid_index]
            label = self.negative_labels[valid_index]
        else:
            # test phase or negative training samples
            sample_index = self.sample_order[index]
            visual_feat = self.visual_feature[sample_index]
            audio_feat = self.audio_feature[sample_index]
            if self.split == 'train':
                label = self.clean_labels[sample_index]
            else:
                # for testing
                label = self.labels[sample_index]

        return visual_feat, audio_feat, label


    def __len__(self):
        if self.split == 'train':
            sample_order = h5py.File(self.sample_order_path, 'r')['order']
            noisy_labels = h5py.File(self.dir_labels_bg_path, 'r')['avadataset']
            length = len(sample_order) + len(noisy_labels)
        elif self.split == 'test':
            sample_order = h5py.File(self.sample_order_path, 'r')['order']
            length = len(sample_order)
        else:
            raise NotImplementedError

        return length

