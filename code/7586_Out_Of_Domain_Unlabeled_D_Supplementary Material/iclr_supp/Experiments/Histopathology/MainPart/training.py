import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
from sklearn.metrics import f1_score, accuracy_score
import argparse
import DatasetEmbedding
import TrainingFlow
import Utils
import ModelFC

argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--train", help="Train data")
argParser.add_argument("-d", "--train_unlabeled", help="Train data unlabeled")
argParser.add_argument("-e", "--test", help="Test data")
argParser.add_argument("-c", "--current_dir", help="Current directory")
argParser.add_argument("-o", "--output", help="Output directory")
argParser.add_argument("-n", "--hpnum", help="Hyper parameter optimization experiment number")
argParser.add_argument("-l", "--labeled_number", help="labeled number")
argParser.add_argument("-m", "--model_path", help="Best model path")
argParser.add_argument("-f", "--load_pretrained", help="Load pretrained or not?")
argParser.add_argument("-s", "--use_scheduler", help="use scheduler or not?")
argParser.add_argument("-w", "--use_weighted_loss", help="use weighted_loss or not?")
argParser.add_argument("-a", "--same_dist_ul", help="same distribution for unlabeled data or not?")
args = argParser.parse_args()


def mylog(text):
    print(f"[MY-LOG] {text}")


def run_model(config=None):

    binaryClass=True
    if config['same_dist_ul']:
        binaryClass=False
    
    # read labeled data
    X_data, y = Utils.read_data(config['input_path'], binaryClass=binaryClass, dataset_type="CRC")
    mylog(f"Data loaded (size:{len(X_data)})")
    X_data_test, y_data_test = Utils.read_data(config['input_test_path'], binaryClass=binaryClass, dataset_type="CRC")
    mylog(f"Test Data loaded (size:{len(X_data_test)})")
    # read unlabeled data
    X_data_ul, y_ul = [], []
    if not config['same_dist_ul']:
        X_data_ul, y_ul = Utils.read_data(config['input_path_ul'], dataset_type="PatchCamelyon")
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
    if not config['same_dist_ul']:
        X_train, y_train = Utils.equal_sampling(X_train, y_train)
        mylog(f"Equal sampling done")
    other_train_x, X_train, other_train_y, y_train = train_test_split(X_train, y_train, test_size=config['labeled_number'], shuffle=True, random_state=42, stratify=y_train)
    mylog(f"Secound part of train spliting")

    has_unlabeled = False
    if config['unlabeled_number'] > 0:
        has_unlabeled = True
        if config['same_dist_ul']:
            if config['unlabeled_number'] > len(other_train_x):
                mylog(f"[Warning] The size of unlabeled data is set to \"{len(other_train_x)}\"")
                _, X_data_ul, _, _ = train_test_split(other_train_x, other_train_y, test_size=len(other_train_x), shuffle=True, random_state=42, stratify=other_train_y)
            else:
                _, X_data_ul, _, _ = train_test_split(other_train_x, other_train_y, test_size=config['unlabeled_number'], shuffle=True, random_state=42, stratify=other_train_y)
        else:
            _, X_data_ul, _, _ = train_test_split(X_data_ul, y_ul, test_size=config['unlabeled_number'], shuffle=True, random_state=42, stratify=y_ul)

    mylog(f"Downsample Labeled (Train):(size:{len(X_train)}), Unlabeled (Train):(size:{len(X_data_ul)})")
    
    config['batch_size_l'] = int(len(X_data)/2)
    config['batch_size_ul'] = int(len(X_data_ul)/2)
    

    y_train, dataloaders_labeled, dataloaders_unlabeled, dataset_sizes = DatasetEmbedding.load_data(config, X_train, y_train, 
                                                                                    X_val, y_val, X_data_test, 
                                                                                    y_data_test, X_data_ul, 
                                                                                    has_unlabeled=has_unlabeled)
    
    model = None
    if config['same_dist_ul']:
        model = ModelFC.myFC(class_num=9)
    else:
        model = ModelFC.myFC(class_num=2)
    if config['load_pretrained']:
        model.load_state_dict(torch.load(config['model_path']))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mylog(f"ML model defined (device: {device})")

    criterion = None
    if config['weighted_loss']:
        class_weights = compute_class_weight('balanced', 
                                            classes=np.unique(y_train), 
                                            y=np.array(y_train))
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(reduction="sum", weight=class_weights)
        mylog("Define weighted CE loss")
    else:
        criterion = nn.CrossEntropyLoss(reduction="sum")
        mylog("Define non-weighted CE loss")

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    mylog("Optimizer (Adam) defined")

    scheduler = None
    if config['scheduler']:
        scheduler = lr_scheduler.LinearLR(optimizer, 
                        start_factor = 1e-8,
                        total_iters = 15)
        mylog("Define scheduler")
    
    model, history = TrainingFlow.train_model(model, dataloaders_labeled, dataloaders_unlabeled, 
                                dataset_sizes, criterion, optimizer, device, scheduler, config=config,
                                num_epochs=config['epoch_num'], has_unlabeled=has_unlabeled)
    
    Utils.save_results(model, history, hp_num, base_path=f"{config['output_path']}")
    mylog("Model run successfully")

    mylog("Start testing")
    y_pred, y_true = Utils.testing(model, dataloaders_labeled['test'], device)
    mylog("testing finished")
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    print("==========")
    mylog(f"Acc: {acc}")
    mylog(f"f1: {f1}")



def main(input_path, input_path_ul, input_path_test, output_path, model_path, 
         labeled_number=1000, unlabeled_number=1000, load_pretrained=True, 
         use_weighted_loss=False, use_scheduler=False, same_dist_ul=True):

    config = {
        'lr': 0.001,
        'weight_decay': 0.001,
        'lamb': 0.001,
        'alpha': 0.001,
        'gamma_l': 0.001,
        'gamma_ul': 0.001,
        'step': 10,
        'labeled_number': labeled_number,
        'unlabeled_number': unlabeled_number,
        'epoch_num': 10,
        'input_path': input_path,
        'input_path_ul': input_path_ul,
        'input_test_path': input_path_test,
        'output_path': output_path,
        'model_path': model_path,
        'load_pretrained': load_pretrained,
        'weighted_loss': use_weighted_loss,
        'scheduler': use_scheduler,
        'same_dist_ul': same_dist_ul
    }
    run_model(config)


hp_num = f"L{args.labeled_number}_UL{args.unlabeled_number}"


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