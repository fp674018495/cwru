# Data science libraries
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# # Pytorch
import torch
# from torch import nn
# from torch.nn import functional as F
# from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
# from torch import optim
# from torch.nn.modules.loss import CrossEntropyLoss

# # Others
# from IPython.core.debugger import set_trace
from pathlib import Path

from helper import get_df_all, download
from train_helper import get_dataloader, fit, validate 
# import nn_model
# from data_urls import URLS


working_dir = Path('.')
DATA_PATH = Path("./data")
save_model_path = working_dir / 'Model'
DE_path = DATA_PATH / '12k/0HP'
random_seed = 7
batch_size = 16

for path in [DATA_PATH, save_model_path]:
    if not path.exists():
        path.mkdir(parents=True)


df_all = get_df_all(DE_path, segment_length=500, normalize=True)
features = df_all.columns[2:]
target = 'label'

X_train, X_valid, y_train, y_valid = train_test_split(df_all[features], 
                                                    df_all[target], 
                                                    test_size=0.20, random_state=random_seed, shuffle=True
                                                    )

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_valid = torch.tensor(X_valid.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_valid = torch.tensor(y_valid.values, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
valid_ds = TensorDataset(X_valid, y_valid)
train_dl, valid_dl = get_dataloader(train_ds, valid_ds, batch_size)
          