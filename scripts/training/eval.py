import os
import argparse
from tqdm import tqdm 

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from models import E2ERegression
from dataset import pad_collate, generate_datasets


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(prog="Regression")
    parser.add_argument("--fe_hidden_dim", type=int, default=64, help="Feature embedding hidden dim")
    parser.add_argument("--fe_output_dim", type=int, default=256, help="Feature embedding output dim")
    parser.add_argument("--se_input_dim", type=int, default=64, help="Sequence embedding input dim")
    parser.add_argument("--se_num_layers", type=int, default=4, help="Sequence embedding num lstm layers")
    parser.add_argument("--load_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset paths, separated with comma(,)")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(0))
        print("Using GPU id {}".format(0))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")

    data_path = args.dataset_path.split(',')
    _, valid_dataset = generate_datasets(data_path, p_val=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=256, num_workers=0, collate_fn=pad_collate)

    input_dim = valid_dataset.input_dim
    model = E2ERegression(
        input_dim,
        args.fe_hidden_dim,
        args.fe_output_dim,
        args.se_input_dim,
        args.se_num_layers
    )

    # model = model.cuda()
    model = model.to(device=device)
    model.load_state_dict(torch.load(args.load_path, map_location=torch.device('cpu')))
    model.eval()
    loss_fn = nn.MSELoss()

    x_delay_all = []
    x_area_all = []
    labels_delay_all = []
    labels_area_all = []
    valid_loss_all = 0
    valid_cnt = 0

    for i, batch in tqdm(enumerate(valid_dataloader)):
        features, sequences, sequences_len, labels = batch
        # features = features.cuda()
        # sequences = sequences.cuda()
        # labels = labels.cuda()
        features = features.to(device=device)
        sequences = sequences.to(device=device)
        labels = labels.to(device=device)
        x = model(features, sequences, sequences_len)

        loss = loss_fn(x, labels)

        x_delay = x[:, 0].cpu().tolist()
        x_area = x[:, 1].cpu().tolist()
        labels_delay = labels[:, 0].cpu().tolist()
        labels_area = labels[:, 1].cpu().tolist()
        x_delay_all += x_delay
        x_area_all += x_area
        labels_delay_all += labels_delay
        labels_area_all += labels_area

        valid_cnt += 1
        valid_loss_all += loss

    model.train()
    corr_delay = stats.spearmanr(x_delay_all, labels_delay_all).correlation
    corr_area = stats.spearmanr(x_area_all, labels_area_all).correlation

    print(f"val_loss {valid_loss_all / valid_cnt}, corr_delay {corr_delay}, corr_area {corr_area}")

