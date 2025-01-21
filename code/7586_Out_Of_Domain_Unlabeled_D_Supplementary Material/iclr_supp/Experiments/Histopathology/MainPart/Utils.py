import torch
import pickle
import os
from tqdm import tqdm
import random
import pickle
import copy
import numpy as np


def read_data(pklfile_path, binaryClass=True, dataset_type="CRC"):
    with open(pklfile_path, 'rb') as f:
        data = pickle.load(f)
    if binaryClass and dataset_type=="CRC":
        data['labels'] = ['TUM' if x == 'TUM' or x == 'STR' else 'NORM' for x in data['labels']]
    return data['embeddings'], data['labels']


def equal_sampling(X_data_org, Y_data_org):
    index = {'TUM': [], 'NORM': []}
    X_data = copy.deepcopy(X_data_org)
    Y_data = copy.deepcopy(Y_data_org)
    for idx, y in enumerate(Y_data):
        index[y].append(idx)
    index['NORM'] = random.choices(index['NORM'], k=len(index['TUM']))
    x_data = []
    y_data = []
    for i, j in zip(index['NORM'], index['TUM']):
        x_data.append(X_data[i])
        x_data.append(X_data[j])
        y_data.extend(['NORM', 'TUM'])
    return np.array(x_data), np.array(y_data)


def save_results(model, history, hp_num, base_path="output"):
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    history_file_path = f"{base_path}/history{hp_num}.pkl"
    with open(history_file_path, 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model_file_path = f"{base_path}/model{hp_num}.pt"
    torch.save(model.state_dict(), model_file_path)


def testing(model, dataloaders, device):
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_pred, y_true