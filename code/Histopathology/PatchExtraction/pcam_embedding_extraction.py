import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import sys
import h5py
import resnet
import pickle
import numpy as np
import shutil
import cv2 as cv
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms
from datetime import datetime
import torch.multiprocessing
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
torch.multiprocessing.set_sharing_strategy('file_system')


def read_data():
    x = {"train": None, "val": None, "test": None}
    y = {"train": None, "val": None, "test": None}

    x['train'] = h5py.File("camelyonpatch_level_2_split_train_x.h5",'r')
    y['train'] = h5py.File("camelyonpatch_level_2_split_train_y.h5",'r')
    x['val'] = h5py.File("camelyonpatch_level_2_split_valid_x.h5",'r')
    y['val'] = h5py.File("camelyonpatch_level_2_split_valid_y.h5",'r')
    x['test'] = h5py.File("camelyonpatch_level_2_split_test_x.h5",'r')
    y['test'] = h5py.File("camelyonpatch_level_2_split_test_y.h5",'r')

    data = {"train": None, "val": None, "test": None}
    for phase in ['train', 'val', 'test']:
        data[phase] = {
            "x": np.squeeze(x[phase]['x']), 
            "y": np.squeeze(y[phase]['y'])
        }
    return data


def save_img(data):
    for phase in ['train', 'val', 'test']:
        os.mkdir(f"dataset/{phase}")
        os.mkdir(f"dataset/{phase}/1")
        os.mkdir(f"dataset/{phase}/0")
        print(f"Save images [{phase}]")
        for i in tqdm(range(len(data[phase]['y']))):
            cv.imwrite(f"dataset/{phase}/{data[phase]['y'][i]}/{i}.png", data[phase]['x'][i, :, :, :])


def main(output_path):
    data = read_data()
    save_img(data)
    del data

    model = resnet.resnet50_trunc_baseline(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def eval_transforms(pretrained=False):
        if pretrained:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
        trnsfrms_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
        return trnsfrms_val

    eval_t = eval_transforms(pretrained=True)

    dataroot = f"dataset/"
    print(f"[My-Log][dataroot]{dataroot}")
    dataset = {}
    dataset_size = {}
    dataloaders = {}
    for phase in ['train', 'val', 'test']:
        dataset[phase] = datasets.ImageFolder(f'dataset/{phase}', transform=eval_t)
        dataset_size[phase] = dataset[phase].__len__()
        dataloaders[phase] = DataLoader(dataset[phase], batch_size=1, shuffle=True)

    for phase in ['train', 'val', 'test']:
        embeddings, labels = [], []
        print(f"Model embedding generation [{phase}]")
        for batch, target in tqdm(dataloaders[phase]):
            with torch.no_grad():
                batch = batch.to(device)
                embeddings.append(model(batch).detach().cpu().numpy())
                labels.append(target.numpy())
        embeddings = np.vstack(embeddings)
        labels = np.vstack(labels).squeeze()

        id2label = dict(map(reversed, dataset[phase].class_to_idx.items()))
        labels = np.array(list(map(id2label.get, labels.ravel())))
        tmp = {'embeddings': embeddings, 'labels': labels}
        with open(f"{phase}.pkl", 'wb') as handle:
            pickle.dump(tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.system(f"cp {phase}.pkl {output_path}")


if __name__ == "__main__":
    os.chdir(sys.argv[1])
    os.mkdir("dataset")
    main(sys.argv[2])