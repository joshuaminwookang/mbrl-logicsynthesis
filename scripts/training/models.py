import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from dataset import SEQ_TO_TOKEN, TOKEN_TO_SEQ

class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        #self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        x = self.dense1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.dense2(x)
        #x = self.dropout(x)
        return x
    
class SequenceEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers):
        super().__init__()
        self.embedding = nn.Embedding(len(SEQ_TO_TOKEN), input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_lstm_layers, batch_first=True)
        
    def forward(self, x, x_len, h):
        x = self.embedding(x)
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, (h, c) = self.lstm(x, (h, h))
        x, x_len = pad_packed_sequence(x, batch_first=True)
        return x, x_len
    
class E2ERegression(nn.Module):
    def __init__(
        self, 
        fe_input_dim, 
        fe_hidden_dim, 
        fe_output_dim,
        se_input_dim,
        se_num_lstm_layers,
        output_dim=2,
    ):
        super().__init__()
        self.fe = FeatureEmbedding(fe_input_dim, fe_hidden_dim, fe_output_dim)
        self.se = SequenceEmbedding(se_input_dim, fe_output_dim, se_num_lstm_layers)
        self.dense = nn.Linear(fe_output_dim, output_dim)
        self.num_lstm_layers = se_num_lstm_layers
        #self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, feature, sequence, sequence_len):
        x = self.fe(feature)
        x = torch.stack([x] * self.num_lstm_layers)
        x, x_len = self.se(sequence, sequence_len, x)
        x = torch.stack([x[i, l - 1, :] for i, l in enumerate(x_len)])
        #x = self.dropout(x)
        x = self.dense(x)
        return x

