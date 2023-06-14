# Data science libraries
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
# # Pytorch
import torch
# from torch import nn
from torch.nn import functional as F
# from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss
from pathlib import Path

from helper import get_df_all, download,awgn
from train_helper import get_dataloader, fit, validate 
from  mymodel.CNN_model import  CNN_1D_2L,CNN_1D_3L
from  mymodel.informer import  Informer
from analysis_data import draw_tsne,draw_tsne_json
from matrix import draw_matrx
import os 
from tqdm import tqdm

print("pid:",os.getpid())
working_dir = Path('.')
DATA_PATH = Path("./data")
save_model_path = working_dir / 'Model2'
PHM_path = DATA_PATH

random_seed = 2
batch_size = 128
d_model= 16
dropout= 0

epochs = 30
lr = 0.00001
wd = 1e-4
betas=(0.99, 0.999)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
number  = 77
num_chanel =1 
input_chanel =7

torch.cuda.manual_seed(number)
torch.manual_seed(number)


def divide_signal(df, segment_length,label,label_y,dic ,idx, seg_num =None ):

    # dic = {}
    # idx = 0
    
    n_sample_points = df.shape[0]
    seg_num = segment_length  if seg_num is None else seg_num
    n_segments = (n_sample_points -segment_length) // seg_num
    
    for segment in range(n_segments):
        dic[idx] = {
            'signal': df[:][seg_num * segment:segment * seg_num+ segment_length].values.reshape(1,1024,input_chanel), 
            # 'label': df.iloc[i,2],
            'label' : label,"label_y":label_y
        }
        idx += 1

    return dic,idx

def divide_label(label):
    if label<120:
        return 0
    elif label < 170:
        return 1
    else:
        return 2


def load_phm(file_name):
    df = pd.read_csv(os.path.join(DATA_PATH,file_name+"_wear.csv"),usecols=["flute_1","flute_2","flute_3"])
    labels = df.max(axis=1) 
    for labels
    dic = {}
    idx = 0
    min_f ,max_f=[0]*7,[0]*7
    for i, file in enumerate(tqdm(os.listdir(os.path.join(DATA_PATH,file_name)))):
        label = divide_label(labels[i])
        label_y = labels[i]
        df = pd.read_csv(os.path.join(DATA_PATH,file_name,file),header=None)
        for i in range(7):
            min_f[i]= min(df.min()[i],min_f[i])
            max_f[i]= max(df.max()[i],max_f[i])
        dic,idx = divide_signal(df, dic =dic,idx=idx,segment_length=1024,seg_num=2000,label=label,label_y=label_y)
    df_tmp = pd.DataFrame.from_dict(dic,orient='index')
    return pd.concat(  [
        df_tmp[['label']],
        (df_tmp[['label_y']]-df_tmp[['label_y']].min())/(df_tmp[['label_y']].max()-df_tmp[['label_y']].min())
    ],
    axis=1,
    ),(np.concatenate(df_tmp["signal"].values)-np.array(min_f))/(np.array(max_f)-np.array(min_f))

Y,X = load_phm("c1")

for path in [DATA_PATH, save_model_path]:
    if not path.exists():
        path.mkdir(parents=True)


X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                    Y, 
                                                    test_size=0.20, random_state=random_seed, shuffle=True
                                                    )

##############  snr ###################
# print("snr")
# x_num = X_train.shape[0]
# for i in tqdm(range(x_num)):
#     ax = awgn(X_train.iloc[i],10)
#     X_train =X_train.append(ax) 
# y_train= pd.concat([y_train,y_train])


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
loss_func = CrossEntropyLoss()


## Instantiate model, optimizer and loss function
########################################################
model =  Informer(
                enc_in=input_chanel,
                dec_in=input_chanel, 
                c_out= 3, 
                seq_len= 512, 
                label_len = 512,
                out_len =10,
                d_model=d_model,
                dropout= dropout
            ).float()

model.to(device)
# from torchsummary import summary
# summary(model,[(1024,1),(1024,1)],depth=4)

opt = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
model, metrics = fit(epochs, model, loss_func, opt, train_dl, valid_dl, train_metric=False)
torch.save(model.state_dict(), save_model_path / 'model16.pth')


# 测试
## Create DataLoader of train and validation set
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)
test_ds = TensorDataset(X_test, y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size)
model =  Informer(
                enc_in=input_chanel,
                dec_in=input_chanel, 
                c_out= 3, 
                seq_len= 512, 
                label_len = 512,
                out_len =10,
                is_test=True
            ).float()
model.to(device)
model.load_state_dict(torch.load(save_model_path / 'model16.pth',map_location=device))
model.eval()
mean_loss, accuracy, (y_true, predictions) = validate(model, test_dl, loss_func)
draw_matrx(y_true, predictions)
print("mean_loss:",mean_loss,"accuracy:", accuracy)
test_YY = draw_tsne_json("res.json")
draw_tsne_json("data/res_cnn.json",test_YY)
draw_tsne_json("data/res_first.json",test_YY)
draw_tsne_json("data/res_informer.json",test_YY)
pass 

