# Data science libraries
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# # Pytorch
import torch
# from torch import nn
from torch.nn import functional as F
# from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss
from pathlib import Path

from helper import get_df_all, download
from train_helper import get_dataloader, fit, validate 
from  mymodel.CNN_model import  CNN_1D_2L,CNN_1D_3L
from  mymodel.informer import  Informer


working_dir = Path('.')
DATA_PATH = Path("./data")
save_model_path = working_dir / 'Model'
DE_path = DATA_PATH / '12k/0HP'
# DE_path = './data/12k/0HP'
random_seed = 7
batch_size = 16
epochs = 100
lr = 0.005
wd = 1e-5
betas=(0.99, 0.999)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



for path in [DATA_PATH, save_model_path]:
    if not path.exists():
        path.mkdir(parents=True)


df_all = get_df_all(DE_path, segment_length=1024,seg_num =512 ,normalize=True)
features = df_all.columns[2:]
target = 'label'

X_train, X_valid, y_train, y_valid = train_test_split(df_all[features], 
                                                    df_all[target], 
                                                    test_size=0.20, random_state=random_seed, shuffle=True
                                                    )
X_test, X_valid, y_test, y_valid = train_test_split(X_valid, 
                                                    y_valid, 
                                                    test_size=0.5, random_state=random_seed, shuffle=True
                                                    )


X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_valid = torch.tensor(X_valid.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_valid = torch.tensor(y_valid.values, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
valid_ds = TensorDataset(X_valid, y_valid)
train_dl, valid_dl = get_dataloader(train_ds, valid_ds, batch_size)

## Instantiate model, optimizer and loss function
model =  Informer(
                enc_in=1,
                dec_in=1, 
                c_out= 10, 
                seq_len= 512, 
                label_len = 512,
                out_len =10,
            ).float()
# model = CNN_1D_3L(len(features))
model.to(device)
opt = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
loss_func = CrossEntropyLoss()

model, metrics = fit(epochs, model, loss_func, opt, train_dl, valid_dl, train_metric=False)

torch.save(model.state_dict(), save_model_path / 'model.pth')


# 测试
## Create DataLoader of train and validation set
# X_test = torch.tensor(X_test.values, dtype=torch.float32)
# y_test = torch.tensor(y_test.values, dtype=torch.float32)
# test_ds = TensorDataset(X_test, y_test)
# DataLoader(test_ds, batch_size=batch_size, shuffle=True)
# model2 = CNN_1D_2L(len(features))
# model2.load_state_dict(torch.load(save_model_path / 'model.pth'))
# model2.eval()
# print(validate(model, valid_dl, loss_func))
