import random
import pickle
from tqdm import tqdm
import numpy as np
import torch
import csv
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

SEQ_TO_TOKEN = {
    '&if -W 300 -K 6 -v': 0, 
    '&st': 1, 
    '&synch2': 2, 
    '&dc2': 3, 
    '&if -W 300 -y -K 6': 4, 
    '&syn2': 5, 
    '&sweep': 6, 
    '&mfs': 7, 
    '&scorr': 8, 
    '&if -W 300 -g -K 6': 9, 
    '&b -d': 10, 
    '&if -W 300 -x -K 6': 11, 
    '&dch': 12, 
    '&b': 13, 
    '&syn4': 14, 
    '&dch -f': 15, 
    '&syn3': 16
}

TOKEN_TO_SEQ = {}
for x, y in SEQ_TO_TOKEN.items():
    TOKEN_TO_SEQ[y] = x

def prepare_dataset(data, feature_dict):
    labels = []
    features = []
    sequences = []
    print('preparing dataset')
    for _, d in tqdm(data.items()):
        label = {}
        label['Path_Delay'] = d['Path_Delay'] / 1e2
        label['Slice_LUTs'] = d['Slice_LUTs'] / 1e3
        labels.append(label)
        name = d['Benchmark']
        assert name in feature_dict
        features.append(feature_dict[name])
        sequences.append(d['Sequence'])
    return features, labels, sequences


def preprocess_sequence(sequences):
    # convert the string representation into a list of tokens
    print('preprocessing sequences')
    seq_list = []
    for seq in tqdm(sequences):
        seq = seq.split(';')[2: -3] # remove the redundant parts
        sl = []
        for s in seq:
            if s.startswith('&'):
                sl.append(SEQ_TO_TOKEN[s])
        seq_list.append(np.array(sl))
    return seq_list

def flatten_all(data):
    flattened_data = []
    for d in data:
        fd = list(d.values())
        flattened_data.append(fd)
    return np.array(flattened_data)

def normalize(data):
    eps = 1e-5
    data_t = np.transpose(data)
    for i in range(len(data_t)):
        mean = np.mean(data_t[i])
        std = np.std(data_t[i])
        data_t[i] = (data_t[i] - mean) / (std + eps)
    return np.transpose(data_t)

def parse_features_csv(feature_dim):
    feature_dict = {}
    with open(f'feature_vectors/med{feature_dim}.csv') as f:
        csvreader = csv.reader(f)
        for i, row in enumerate(csvreader):
            if i == 0: continue
            name, features = row[0], row[1:]
            feature_dict[name] = [float(x) for x in features]
    return feature_dict

def preprocess_data(data_path, feature_dim):
    if not isinstance(data_path, list):
        data_path = [data_path]
    features, labels, sequences = [], [], []

    feature_dict = parse_features_csv(feature_dim)

    for _data_path in data_path:
        with open(_data_path, 'rb') as f:
            data = pickle.load(f)

        _features, _labels, _sequences = prepare_dataset(data, feature_dict)
        features += _features
        labels += _labels
        sequences += _sequences

    sequences_list = preprocess_sequence(sequences)
    labels_flattened = flatten_all(labels)

    features_normalized = np.array(features)
    # Uncomment the following for testing/ablation
    #features_normalized = np.zeros_like(features_normalized)
    #features_normalized = np.random.rand(*features_normalized.shape)

    return features_normalized, np.array(sequences_list), labels_flattened

class CustomDataset(Dataset):
    def __init__(self, features=None, sequences=None, labels=None, data_path=None):
        if data_path is not None:
            self.features, self.sequences, self.labels = preprocess_data(data_path)
        else:
            assert features is not None 
            assert sequences is not None 
            assert labels is not None 
            self.features, self.sequences, self.labels = features, sequences, labels
        self.input_dim = self.features.shape[-1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx]).to(torch.float)
        sequence = torch.tensor(self.sequences[idx]).to(torch.long)
        label = torch.tensor(self.labels[idx]).to(torch.float)
        #return {'feature': feature, 'sequence': sequence, 'label': label}
        return feature, sequence, label

def generate_datasets(data_path, p_val=0.2, feature_dim=64):
    features, sequences, labels = preprocess_data(data_path, feature_dim=feature_dim)
    num_training_sets = int((1 - p_val) * len(features))
    indices_all = np.array(list(range(len(features))))
    random.seed(100)
    random.shuffle(indices_all)
    print(indices_all)
    train_indices = indices_all[:num_training_sets]
    val_indices = indices_all[num_training_sets:]

    train_dataset = CustomDataset(
        features[train_indices], 
        sequences[train_indices],
        labels[train_indices],
    )
    valid_dataset = CustomDataset(
        features[val_indices], 
        sequences[val_indices],
        labels[val_indices],
    )

    return train_dataset, valid_dataset

def pad_collate(batch):
    features, sequences, labels, = zip(*batch)
    sequences_len = [len(s) for s in sequences]
    features_pad = pad_sequence(features, batch_first=True, padding_value=0)
    sequences_pad = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=0) 
    return features_pad, sequences_pad, sequences_len, labels_pad


if __name__ == '__main__': 
    features, sequences, labels = preprocess_data('../../epfl_arithmetic.pkl')
