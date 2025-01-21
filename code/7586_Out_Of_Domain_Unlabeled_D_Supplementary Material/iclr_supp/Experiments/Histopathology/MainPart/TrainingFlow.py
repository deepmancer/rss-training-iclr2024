import torch
import copy
from tqdm import tqdm


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

    for epoch in range(num_epochs):
        print(f"==========\nEpoch {epoch+1}/{num_epochs}\n==========")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                if has_unlabeled:
                    for (x_l_org, y), x_ul_org in tqdm(zip(dataloaders_labeled[phase], dataloaders_unlabeled[phase])):
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
                    for x_l_org, y in tqdm(dataloaders_labeled[phase]):
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
                for x_l_org, y in tqdm(dataloaders_labeled[phase]):
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
            print(f"{phase}_loss: {epoch_loss:.4f}\n{phase}_acc: {epoch_acc:.4f}")
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history