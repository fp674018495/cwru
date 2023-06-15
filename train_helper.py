

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss
import json
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm
# from one_cycle import OneCycle, update_lr, update_mom


# Functions for training
def get_dataloader(train_ds, valid_ds, bs):

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def loss_batch(model, loss_func, xb, yb, opt=None):

    out = model(xb)
    # out = model(xb,xb)
    loss = loss_func(out, yb)
    pred = torch.argmax(out, dim=1).cpu().numpy()

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb), pred

def loss_batch1(model, loss_func, xb, yb, opt=None):

    # out = model(xb)
    out = model(xb)
    loss = loss_func(out, yb)
    pred = torch.argmax(out, dim=1).cpu().numpy()

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    with open("res.json",mode="a+") as fp:
        json.dump(out.cpu().detach().numpy().tolist(),fp)
        json.dump(yb.cpu().detach().numpy().tolist(),fp)
        fp.write("\n")
    return loss.item(), len(xb), pred


def loss_batch2(model, loss_func, loss_func2,xb,  y_c,y_r, opt=None):

    out = model(xb)

    loss = loss_func(out[0], y_c) +loss_func2(out[1], y_r)
    
    pred = torch.argmax(out[0], dim=1).cpu().numpy()

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb), pred



def fit(epochs, model, loss_func,loss_func2, opt, train_dl, valid_dl, one_cycle=None, train_metric=False):

    print(
        'EPOCH', '\t', 
        'Train Loss', '\t',
        'Val Loss', '\t', 
        'Train Acc', '\t',
        'Val Acc', '\t')
    # Initialize dic to store metrics for each epoch.
    metrics_dic = {}
    metrics_dic['train_loss'] = []
    metrics_dic['train_accuracy'] = []
    metrics_dic['val_loss'] = []
    metrics_dic['val_accuracy'] = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    best_loss = 0.9
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        num_examples = 0
        for xb, y_c,y_r in tqdm(train_dl):
            xb, y_c,y_r = xb.to(device), y_c.to(device), y_r.to(device)
            loss, batch_size, pred = loss_batch2(model, loss_func,loss_func2, xb,  y_c,y_r, opt)
            if train_metric == False:
                train_loss += loss
                num_examples += batch_size
            
            if one_cycle:
                lr, mom = one_cycle.calc()
                update_lr(opt, lr)
                update_mom(opt, mom)

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy, _ = validate2(model, valid_dl, loss_func,loss_func2)
            if train_metric:
                train_loss, train_accuracy, _ = validate2(model, train_dl, loss_func,loss_func2)
            else:
                train_loss = train_loss / num_examples

        metrics_dic['val_loss'].append(val_loss)
        metrics_dic['val_accuracy'].append(val_accuracy)
        metrics_dic['train_loss'].append(train_loss)
        metrics_dic['train_accuracy'].append(train_accuracy)
        
        if  best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'Model/{val_accuracy}.pth')
        print(
            f'{epoch} \t', 
            f'{train_loss:.05f}', '\t',
            f'{val_loss:.05f}', '\t', 
            f'{train_accuracy:.05f}', '\t'
            f'{val_accuracy:.05f}', '\t')
        
    metrics = pd.DataFrame.from_dict(metrics_dic)

    return model, metrics

def validate(model, dl, loss_func):
    total_loss = 0.0
    total_size = 0
    predictions = []
    y_true = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for xb, yb in dl: 
        xb, yb = xb.to(device), yb.to(device)
        loss, batch_size, pred = loss_batch1(model, loss_func, xb, yb)
        total_loss += loss*batch_size
        total_size += batch_size
        predictions.append(pred)
        y_true.append(yb.cpu().numpy())
    mean_loss = total_loss / total_size
    predictions = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    accuracy = np.mean((predictions == y_true))
    return mean_loss, accuracy, (y_true, predictions)


def validate2(model, dl, loss_func,loss_func2):
    total_loss = 0.0
    total_size = 0
    predictions = []
    y_true = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for xb, y_c,y_r in dl: 
        xb, y_c,y_r = xb.to(device), y_c.to(device), y_r.to(device)
        loss, batch_size, pred = loss_batch2(model, loss_func,loss_func2, xb,  y_c,y_r)
        total_loss += loss*batch_size
        total_size += batch_size
        predictions.append(pred)
        y_true.append(y_c.cpu().numpy())
    mean_loss = total_loss / total_size
    predictions = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    accuracy = np.mean((predictions == y_true))
    return mean_loss, accuracy, (y_true, predictions)