import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import pickle
import os
import copy
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import wandb
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--train", help="Train data")
argParser.add_argument("-d", "--train_unlabeled", help="Train data unlabeled")
argParser.add_argument("-e", "--test", help="Test data")
argParser.add_argument("-c", "--current_dir", help="Current directory")
argParser.add_argument("-o", "--output", help="Output directory")
argParser.add_argument("-n", "--hpnum", help="Hyper parameter optimization experiment number")
argParser.add_argument("-u", "--unlabeled_number", help="Unlabeled number")
argParser.add_argument("-l", "--labeled_number", help="labeled number")
argParser.add_argument("-m", "--model_path", help="Best model path")
argParser.add_argument("-f", "--load_pretrained", help="Load pretrained or not?")
argParser.add_argument("-s", "--use_scheduler", help="use scheduler or not?")
argParser.add_argument("-w", "--use_weighted_loss", help="use weighted_loss or not?")
argParser.add_argument("-a", "--same_dist_ul", help="same distribution for unlabeled data or not?")
args = argParser.parse_args()

################################################
################################################
##        Replace your wandb token below      ##
################################################
################################################
wandb.login(key="<your-token>")
################################################
################################################

def mylog(text):
    print(f"[MY-LOG] {text}")


class EmbeddingDataset(Dataset):
    def __init__(self, X_data, labels):
        self.embeddings = X_data
        self.labels = np.unique(labels, return_inverse=True)[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding_vector = torch.tensor(self.embeddings[idx])
        label_onehot = torch.tensor(self.labels[idx])
        return embedding_vector, label_onehot


class EmbeddingDataset_ul(Dataset):
    def __init__(self, X_data):
        self.embeddings = X_data

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding_vector = torch.tensor(self.embeddings[idx])
        return embedding_vector


class myFC(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=[2048, 2048], class_num=2):
        super(myFC, self).__init__()
        self.fc1 = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim[1], class_num)
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.softmax(x)
        return x


def read_data(pklfile_path, binaryClass=True, dataset_type="CRC"):
    mylog(f"Loading data from {pklfile_path}")
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


def load_data(config, X_train, y_train, X_val, y_val, X_test, y_test, X_data_ul, has_unlabeled=True):

    dataset = {'train': EmbeddingDataset(X_train, y_train),
              'val': EmbeddingDataset(X_val, y_val),
              'test': EmbeddingDataset(X_test, y_test)}
    mylog("datasets created")

    dataloaders_labeled = {x: DataLoader(dataset[x], batch_size=config['batch_size_l'], num_workers=1) for x in ['train', 'val', 'test']}
    mylog("dataloaders created")
    dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val', 'test']}
    mylog("dataset_sizes created")

    if has_unlabeled:
        dataset_ul = {'train': EmbeddingDataset_ul(X_data_ul)}
        mylog("datasets unlabeld created")

        dataloaders_unlabeled = {x: DataLoader(dataset_ul['train'], batch_size=config['batch_size_ul'], num_workers=1) for x in ['train']}
        mylog("dataloaders created")

        return y_train, dataloaders_labeled, dataloaders_unlabeled, dataset_sizes
    else:
        return y_train, dataloaders_labeled, None, dataset_sizes


def optimize_x_adv(model, x_org, y, step, alpha, gamma, criterion):
    model.eval()
    x_adv = copy.deepcopy(x_org)
    with torch.set_grad_enabled(True):
        x_adv.requires_grad = True
        for t in range(step):
            pred = model(x_adv)
            loss = criterion(pred, y)
            cost = torch.sum(torch.norm((x_adv-x_org), dim=1)**2)
            phi = loss - gamma * cost
            grad = torch.autograd.grad(phi, x_adv)
            lr = alpha / (t + 1)
            x_adv = x_adv + lr * grad[0]
    return x_adv


def train_model(model, dataloaders_labeled, dataloaders_unlabeled, dataset_sizes,  
                criterion, optimizer, device, scheduler, config, num_epochs=25, has_unlabeled=True):
    history = {'train_loss': [], 'val_loss': [], 
                    'train_acc': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                if has_unlabeled:
                    for (x_l_org, y), x_ul_org in zip(dataloaders_labeled[phase], dataloaders_unlabeled[phase]):
                        x_l_org = x_l_org.to(device)
                        x_ul_org = x_ul_org.to(device)
                        y = y.to(device)

                        optimizer.zero_grad()
                        preds = None
                        
                        with torch.set_grad_enabled(phase == 'train'):
                            # labeled data
                            x_adv = optimize_x_adv(model=model, x_org=x_l_org, y=y, 
                                                step=config['step'], alpha=config['alpha'], 
                                                gamma=config['gamma_l'], criterion=criterion)
                            pred = model(x_adv)
                            loss = criterion(pred, y)
                            cost = torch.sum(torch.norm((x_adv-x_l_org), dim=1)**2)
                            phi = (loss - config['gamma_l'] * cost)/len(x_l_org)
                            _, preds = torch.max(pred, 1)
                        
                            # unlabeled
                            pred_org = model(x_ul_org)
                            x_adv = optimize_x_adv(model=model, x_org=x_ul_org, y=pred_org, 
                                                step=config['step'], alpha=config['alpha'], 
                                                gamma=config['gamma_ul'], criterion=criterion)
                            pred = model(x_adv)
                            loss = criterion(pred, pred_org)
                            cost = torch.sum(torch.norm((x_adv-x_ul_org), dim=1)**2)
                            phi_ul = (loss - config['gamma_ul'] * cost)/len(x_ul_org)

                            loss = (phi + config['lamb']*phi_ul)*100

                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item()
                        running_corrects += torch.sum(preds == y.data)
                else:
                    for x_l_org, y in dataloaders_labeled[phase]:
                        x_l_org = x_l_org.to(device)
                        y = y.to(device)

                        optimizer.zero_grad()
                        preds = None
                        
                        with torch.set_grad_enabled(phase == 'train'):
                            # labeled data
                            x_adv = optimize_x_adv(model=model, x_org=x_l_org, y=y, 
                                                step=config['step'], alpha=config['alpha'], 
                                                gamma=config['gamma_l'], criterion=criterion)
                            pred = model(x_adv)
                            loss = criterion(pred, y)
                            cost = torch.sum(torch.norm((x_adv-x_l_org), dim=1)**2)
                            phi = (loss - config['gamma_l'] * cost)/len(x_l_org)
                            _, preds = torch.max(pred, 1)

                            loss = phi

                            loss.backward()
                            optimizer.step()

                        # statistics
                        running_loss += loss.item() * x_l_org.size(0)
                        running_corrects += torch.sum(preds == y.data)
            else:
                for x_l_org, y in dataloaders_labeled[phase]:
                    x_l_org = x_l_org.to(device)
                    y = y.to(device)
                    with torch.no_grad():
                        pred = model(x_l_org)
                        loss = criterion(pred, y)
                        _, preds = torch.max(pred, 1)
                    # statistics
                    running_loss += loss.item()
                    # Due to the fact that in the Train part, once divided by the number, 
                    # once multiplied by the number, we do not multiply here.
                    running_corrects += torch.sum(preds == y.data)
            
            if phase == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            history[f'{phase}_loss'].append(epoch_loss)
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            history[f'{phase}_acc'].append(float(epoch_acc.cpu().numpy()))

            tmp = {f'{phase}_loss': epoch_loss, 
                    f'{phase}_acc': epoch_acc,
                    'lr': config['lr'],
                    'epoch': epoch}
            wandb.log(tmp)
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def save_results(model, history, base_path="output"):
    global ex_num, total_ex_num, hp_num
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    history_file_path = f"{base_path}/history.pkl"
    with open(history_file_path, 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[Log] history has been saved in \"{history_file_path}\"')

    model_file_path = f"{base_path}/model_hp{hp_num}_{ex_num}.pt"
    torch.save(model.state_dict(), model_file_path)
    print(f'[Log] model has been saved in \"{model_file_path}\"')


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
    mylog("Testing loop finished")
    return y_pred, y_true


def run_model(config=None):
    global ex_num, total_ex_num, hp_num, proj_name, tags
    
    with wandb.init(entity='<your-entity-name>', project=proj_name, name=f"{hp_num}_{ex_num}", 
                    config=config, tags=tags):
        config_wandb = wandb.config
        myconfig = {
            'batch_size_l': 0,
            'batch_size_ul': 0,
            'lr': config_wandb.lr,
            'input_path': config_wandb.input_path,
            'input_path_ul': config_wandb.input_path_ul,
            'input_test_path': config_wandb.input_test_path,
            'epoch_num': config_wandb.epoch_num,
            'weight_decay': config_wandb.weight_decay,
            'model_path': config_wandb.model_path,
            'output_path': config_wandb.output_path,
            'gamma_l': config_wandb.gamma_l,
            'gamma_ul': config_wandb.gamma_ul,
            'lamb': config_wandb.lamb,
            'alpha': config_wandb.alpha,
            'step': config_wandb.step,
            'labeled_number': config_wandb.labeled_number,
            'unlabeled_number': config_wandb.unlabeled_number,
            'load_pretrained': config_wandb.load_pretrained,
            'weighted_loss': config_wandb.weighted_loss,
            'scheduler': config_wandb.scheduler,
            'same_dist_ul': config_wandb.same_dist_ul
        }

        print("="*10)
        ex_num+=1
        mylog(f"[Ex.{ex_num}/{total_ex_num}]")
        print("="*10)

        binaryClass=True
        if myconfig['same_dist_ul']:
            binaryClass=False
        
        # read labeled data
        X_data, y = read_data(myconfig['input_path'], binaryClass=binaryClass, dataset_type="CRC")
        mylog(f"Data loaded (size:{len(X_data)})")
        X_data_test, y_data_test = read_data(myconfig['input_test_path'], binaryClass=binaryClass, dataset_type="CRC")
        mylog(f"Test Data loaded (size:{len(X_data_test)})")
        # read unlabeled data
        X_data_ul, y_ul = [], []
        if not myconfig['same_dist_ul']:
            X_data_ul, y_ul = read_data(myconfig['input_path_ul'], dataset_type="PatchCamelyon")
            mylog(f"Data loaded (size:{len(X_data_ul)})")

        # spliting
        X_train, X_val, y_train, y_val = train_test_split(X_data, y, test_size=0.2, 
                                                shuffle=True, random_state=42, 
                                                stratify=y)
        ـ, X_val, ـ, y_val = train_test_split(X_val, y_val, test_size=0.7, 
                                                shuffle=True, random_state=42, 
                                                stratify=y_val)
        mylog(f"First part of train test spliting")

        # equal_sampling and downsampleing
        if not myconfig['same_dist_ul']:
            X_train, y_train = equal_sampling(X_train, y_train)
            mylog(f"Equal sampling done")
        other_train_x, X_train, other_train_y, y_train = train_test_split(X_train, y_train, test_size=myconfig['labeled_number'], shuffle=True, random_state=42, stratify=y_train)
        mylog(f"Secound part of train spliting")

        has_unlabeled = False
        if myconfig['unlabeled_number'] > 0:
            has_unlabeled = True
            if myconfig['same_dist_ul']:
                if myconfig['unlabeled_number'] > len(other_train_x):
                    mylog(f"[Warning] The size of unlabeled data is set to \"{len(other_train_x)}\"")
                    _, X_data_ul, _, _ = train_test_split(other_train_x, other_train_y, test_size=len(other_train_x), shuffle=True, random_state=42, stratify=other_train_y)
                else:
                    _, X_data_ul, _, _ = train_test_split(other_train_x, other_train_y, test_size=myconfig['unlabeled_number'], shuffle=True, random_state=42, stratify=other_train_y)
            else:
                _, X_data_ul, _, _ = train_test_split(X_data_ul, y_ul, test_size=myconfig['unlabeled_number'], shuffle=True, random_state=42, stratify=y_ul)

        mylog(f"Downsample Labeled (Train):(size:{len(X_train)}), Unlabeled (Train):(size:{len(X_data_ul)})")
        wandb.log({"labeled_data_size": len(X_train), "Unlabeled_data_size": len(X_data_ul)})
        
        myconfig['batch_size_l'] = int(len(X_data)/2)
        myconfig['batch_size_ul'] = int(len(X_data_ul)/2)
        

        y_train, dataloaders_labeled, dataloaders_unlabeled, dataset_sizes = load_data(myconfig, X_train, y_train, 
                                                                                       X_val, y_val, X_data_test, 
                                                                                       y_data_test, X_data_ul, 
                                                                                       has_unlabeled=has_unlabeled)
        
        model = None
        if myconfig['same_dist_ul']:
            model = myFC(class_num=9)
        else:
            model = myFC(class_num=2)
        if myconfig['load_pretrained']:
            model.load_state_dict(torch.load(myconfig['model_path']))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        mylog(f"ML model defined (device: {device})")

        criterion = None
        if myconfig['weighted_loss']:
            class_weights = compute_class_weight('balanced', 
                                                classes=np.unique(y_train), 
                                                y=np.array(y_train))
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

            criterion = nn.CrossEntropyLoss(reduction="sum", weight=class_weights)
            mylog("Define weighted CE loss")
        else:
            criterion = nn.CrossEntropyLoss(reduction="sum")
            mylog("Define non-weighted CE loss")

        optimizer = optim.Adam(model.parameters(), lr=myconfig['lr'], weight_decay=myconfig['weight_decay'])
        mylog("Optimizer (Adam) defined")

        scheduler = None
        if myconfig['scheduler']:
            scheduler = lr_scheduler.LinearLR(optimizer, 
                         start_factor = 1e-8,
                         total_iters = 15)
            mylog("Define scheduler")
        
        model, history = train_model(model, dataloaders_labeled, dataloaders_unlabeled, 
                                    dataset_sizes, criterion, optimizer, device, scheduler, config=myconfig,
                                    num_epochs=myconfig['epoch_num'], has_unlabeled=has_unlabeled)
        
        save_results(model, history, base_path=f"{myconfig['output_path']}")
        mylog("Model run successfully")

        mylog("Start testing")
        y_pred, y_true = testing(model, dataloaders_labeled['test'], device)
        mylog("testing finished")
        f1 = f1_score(y_true, y_pred, average='macro')
        mylog("F1 done")
        acc = accuracy_score(y_true, y_pred)
        mylog("acc done")
        wandb.sklearn.plot_confusion_matrix(y_true, 
                                            y_pred, 
                                            np.unique(y))
        mylog("Metric calculated")
        tmp = {
            "test_acc": acc, 
            "test_f1": f1,
            "test_size": dataset_sizes['test']
        }
        wandb.log(tmp)
        mylog("Run finished")



def main(input_path, input_path_ul, input_path_test, output_path, model_path, 
         labeled_number=1000, unlabeled_number=1000, load_pretrained=True, 
         use_weighted_loss=False, use_scheduler=False, same_dist_ul=True):
    global ex_num, total_ex_num, proj_name
    print("Main method running ...")
    
    sweep_config = {'method': 'random'}

    metric = {'name': 'val_acc', 
            'goal': 'maximize'}

    sweep_config['metric'] = metric

    hyper_parameters = {
        'lr': { 'values': [float(f"1e-{i}") for i in range(1,5)]},
        'weight_decay': { 'values': [float(f"1e-{i}") for i in range(2,7)]},
        'lamb': { 'values': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
        'alpha': { 'values': [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
        'gamma_l': { 'values': [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]},
        'gamma_ul': { 'values': [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]},
        'step': {'value': 10},
        'labeled_number': {'value': labeled_number},
        'unlabeled_number': {'value': unlabeled_number},
        'epoch_num': {'value': 10},
        'input_path': {'value': input_path},
        'input_path_ul': {'value': input_path_ul},
        'input_test_path': {'value': input_path_test},
        'output_path': {'value': output_path},
        'model_path': {'value': model_path},
        'load_pretrained': {'value': load_pretrained},
        'weighted_loss': {'value': use_weighted_loss},
        'scheduler': {'value': use_scheduler},
        'same_dist_ul': {'value': same_dist_ul}
    }
    sweep_config['parameters'] = hyper_parameters
    sweep_id = wandb.sweep(sweep_config, project=proj_name)
    wandb.agent(sweep_id, run_model, count=total_ex_num)


ex_num = 0
total_ex_num = 100
hp_num = f"L{args.labeled_number}_UL{args.unlabeled_number}"
proj_name = "ssdrl-testing-final"
tags = [f"L{args.labeled_number}", f"UL{args.unlabeled_number}"]


if __name__ == "__main__":
    load_pretrained = False
    if args.load_pretrained == 'True' or args.load_pretrained == 'true' or args.load_pretrained == 't' or args.load_pretrained == 'T':
        load_pretrained = True
    
    use_weighted_loss = False
    if args.use_weighted_loss == 'True' or args.use_weighted_loss == 'true' or args.use_weighted_loss == 't' or args.use_weighted_loss == 'T':
        use_weighted_loss = True
    
    use_scheduler = False
    if args.use_scheduler == 'True' or args.use_scheduler == 'true' or args.use_scheduler == 't' or args.use_scheduler == 'T':
        use_scheduler = True
    
    same_dist_ul = False
    if args.same_dist_ul == 'True' or args.same_dist_ul == 'true' or args.same_dist_ul == 't' or args.same_dist_ul == 'T':
        same_dist_ul = True

    mylog(f"[args] train: {args.train}")
    mylog(f"[args] test: {args.test}")
    mylog(f"[args] output: {args.output}")
    mylog(f"[args] current_dir: {args.current_dir}")
    mylog(f"[args] hp_num: {args.hpnum}")
    mylog(f"[args] labeled_number: {args.labeled_number}")
    mylog(f"[args] unlabeled_number: {args.unlabeled_number}")
    mylog(f"[args] model_path: {args.model_path}")
    mylog(f"[args] load_pretrained: {load_pretrained}")
    mylog(f"[args] use_scheduler: {use_scheduler}")
    mylog(f"[args] use_weighted_loss: {use_weighted_loss}")
    mylog(f"[args] same_dist_ul: {same_dist_ul}")
    os.chdir(f"{args.current_dir}")

    main(input_path=args.train, 
         input_path_ul=args.train_unlabeled, 
         input_path_test=args.test, 
         output_path=args.output, 
         model_path=args.model_path, 
         unlabeled_number=int(args.unlabeled_number), 
         labeled_number=int(args.labeled_number), 
         load_pretrained=load_pretrained,
         use_weighted_loss=use_weighted_loss,
         use_scheduler=use_scheduler,
         same_dist_ul=same_dist_ul
    )