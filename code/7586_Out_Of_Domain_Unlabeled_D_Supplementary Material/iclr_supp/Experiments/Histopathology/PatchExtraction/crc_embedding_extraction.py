import os
import sys
import pickle
import resnet
import numpy as np
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms
from datetime import datetime
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def main(argv):
    input_data_path = argv[1]
    output_embedding_path = argv[2]
    output_file_name = argv[3]

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
    
    dataroot = f"{input_data_path}/"
    print(f"[My-Log][dataroot]{dataroot}")
    dataset = torchvision.datasets.ImageFolder(dataroot, transform=eval_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    embeddings, labels = [], []
    for batch, target in tqdm(dataloader):
        with torch.no_grad():
            batch = batch.to(device)
            embeddings.append(model(batch).detach().cpu().numpy())
            labels.append(target.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels).squeeze()

    id2label = dict(map(reversed, dataset.class_to_idx.items()))
    labels = np.array(list(map(id2label.get, labels.ravel())))

    asset_dict = {'embeddings': embeddings, 'labels': labels}

    ts = int(datetime.timestamp(datetime.now()))
    with open(f"{output_embedding_path}/{output_file_name}_{ts}.pkl", 'wb') as handle:
        pickle.dump(asset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(sys.argv)