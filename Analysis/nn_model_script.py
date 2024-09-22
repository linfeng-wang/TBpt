#%%
from array import array
from cmath import nan
from pyexpat import model
import statistics
from tkinter.ttk import Separator
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
from torch.autograd import variable
from itertools import chain
from sklearn import metrics as met
import pickle
from icecream import ic

import matplotlib.pyplot as plt
import pathlib
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from importlib import reload
# import util
# import model_torch_simple
# from torchmetrics import Accuracy
from tqdm import tqdm
import argparse
from icecream import ic
import numpy as np
from PIL import Image
device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 42
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

df_imputed = pd.read_csv('/mnt/storageG1/lwang/Projects/TBpt/Analysis/organised_data_mean_imput.csv')

y = df_imputed[['outcome']]
X = df_imputed[['age_of_onset', 'gender', 'country', 'employment', 'type_of_resistance',
       'bmi', 'lung_localization', 'overall_percent_of_abnormal_volume',
       'pleural_effusion_percent_of_hemithorax_involved',
       'ispleuraleffusionbilateral', 'other_non_tb_abnormalities',
       'are_mediastinal_lymphnodes_present', 'collapse', 'smallcavities',
       'mediumcavities', 'largecavities',
       'isanylargecavitybelongtoamultisextantcavity',
       'canmultiplecavitiesbeseen', 'infiltrate_lowgroundglassdensity',
       'infiltrate_mediumdensity', 'infiltrate_highdensity', 'smallnodules',
       'mediumnodules', 'largenodules', 'hugenodules',
       'isanycalcifiedorpartiallycalcifiednoduleexist',
       'isanynoncalcifiednoduleexist', 'isanyclusterednoduleexists',
       'aremultiplenoduleexists', 'lowgroundglassdensityactivefreshnodules',
       'mediumdensitystabalizedfibroticnodules',
       'highdensitycalcifiedtypicallysequella', 'period_span', 'regimen_count',
       'Others', 'cytostatics',
       'Systemically administered glucocorticoids', 'Psychiatric illness',
       'Pneumoconiosis', 'Anemia', 'Hepatic diseases', 'Renal disease',
       'Diabetes', 'Post-COVID-19', 'COVID-19', 'HIV',
       'Amoxicillin-clavulanate', 'Antiretroviral therapy', 'Bedaquiline',
       'Capreomycin', 'Clarithromycin', 'Clofazimine',
       'Cotrimoxazol preventive', 'Cycloserine', 'Delamanid', 'Ethambutol',
       'Ethionamide', 'Fluoroquinolones', 'Gatifloxacin',
       'Imipenem-cilastatin', 'Isoniazid', 'Kanamycin', 'Levofloxacin',
       'Linezolid', 'Moxifloxacin', 'Mycobutin', 'Ofloxacin', 'Pretomanid',
       'Prothionamide', 'Pyrazinamide', 'Rifampicin', 'Streptomycin',
       'Terizidone', 'p-aminosalicylic acid', 'Amikacin',
       'Aminoglycosides - injectible agents']]



train_data, test_data, train_target, test_target = train_test_split(X, y,test_size=.2,random_state =123, stratify=y)


N_samples = train_data.shape[0]
DRUGS = ['AMIKACIN',
 'CAPREOMYCIN',
 'CIPROFLOXACIN',
 'ETHAMBUTOL',
 'ETHIONAMIDE',
 'ISONIAZID',
 'KANAMYCIN',
 'LEVOFLOXACIN',
 'MOXIFLOXACIN',
 'OFLOXACIN',
 'PYRAZINAMIDE',
 'RIFAMPICIN',
 'STREPTOMYCIN']

# DRUGS = ['ISONIAZID']

DRUGS = train_target.columns
LOCI = train_data.columns
assert set(DRUGS) == set(train_target.columns)
N_drugs = len(DRUGS)


def get_masked_loss(loss_fn):
    """
    Returns a loss function that ignores NaN values
    """

    def masked_loss(y_true, y_pred):
        y_pred = y_pred.view(-1, 1)  # Ensure y_pred has the same shape as y_true and non_nan_mask
        # ic(y_true)
        non_nan_mask = ~y_true.isnan()
        # ic(non_nan_mask)
        y_true_non_nan = y_true[non_nan_mask]
        y_pred_non_nan = y_pred[non_nan_mask]

        return loss_fn(y_pred_non_nan, y_true_non_nan)

    return masked_loss

masked_MSE = get_masked_loss(torch.nn.MSELoss())

class OneHotSeqsDataset(torch.utils.data.Dataset): #? what's the difference between using inheritance and not?
    def __init__(
        self,
        X_df,
        y_df,
    ):
        self.X_df = X_df
        self.y_df = y_df
        if not self.X_df.index.equals(self.y_df.index):
            raise ValueError(
                "Indices of sequence and yistance dataframes don't match up"
            )

    def __getitem__(self, index):
        """
        numerical index --> get `index`-th sample
        string index --> get sample with name `index`
        """
        X_entry = self.X_df.iloc[index]
        y_entry = self.y_df.iloc[index]


        # if self.transform:
            # y = np.log2(y)
            
            # self.y_mean = self.y_df.mean()
            # self.y_std = self.y_df.std()
            # y = (y - self.y_mean) / self.y_std
            # y = self.transform(y)
        return torch.tensor(X_entry), torch.tensor(y_entry).long().flatten().squeeze()

    def __len__(self):
        return self.y_df.shape[0]

training_dataset = OneHotSeqsDataset(train_data, train_target)
train_dataset, val_dataset = random_split(training_dataset, [int(len(training_dataset)*0.8), len(training_dataset)-int(len(training_dataset)*0.8)])

torch.cuda.empty_cache()

class Original_Model(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_classes=5,
        num_filters=64,
        filter_length=12,
        num_conv_layers=2,
        filter_scaling_factor=1,  # New parameter
        num_dense_neurons=256,
        num_dense_layers=2,
        conv_dropout_rate=0.0,
        dense_dropout_rate=0.2,
        input_dim = 76,
        return_logits=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.conv_dropout_rate = conv_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.return_logits = return_logits
        self.input_dim = input_dim
        
        # now define the actual model
        self.feature_extraction_layer = nn.Sequential(
            self._conv_layer(self.in_channels, self.num_filters, self.filter_length))
            # self._conv_layer(self.num_filters, self.num_filters, self.filter_length))
            
        self.max_l = nn.MaxPool1d(3, stride=1)
        
        #dynamic filter scaling from deepram
        # current_num_filters = num_filters
        current_num_filters = 128
        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layers):
            layer = self._conv_layer(self.num_filters, int(current_num_filters * filter_scaling_factor), 3)
            self.conv_layers.append(layer)
            current_num_filters = int(current_num_filters * filter_scaling_factor)
        # now define the actual model
        # self.feature_abstraction_layer = nn.Sequential(
        #     self._conv_layer(self.num_filters, 32, self.filter_length)
        #     self._conv_layer(32,  32, self.filter_length))
        
        self.dense_layers = nn.ModuleList(
            self._dense_layer(in_, num_dense_neurons)
            for in_ in [self.input_dim]
            + [num_dense_neurons] * (self.num_dense_layers - 1) #how does this work?
        )
        
        # self.dense_layers = nn.Sequential(
        #     self._dense_layer(110240, num_dense_neurons),
        #     self._dense_layer(num_dense_neurons, num_dense_neurons))
        
        self.prediction_layer = (
            nn.Linear(num_dense_neurons, num_classes)
            if return_logits
            else nn.Sequential(nn.Linear(num_dense_neurons, num_classes), nn.SELU()) #difference between sequential and nn.moduleList?
        )

        self.apply(self.init_weights)    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)        
        
    def _conv_layer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Dropout(p=self.conv_dropout_rate),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.SELU(),
        )

    def _dense_layer(self, n_in, n_out):
        return nn.Sequential(
            nn.Linear(n_in, n_out),
            # nn.Dropout(p=self.dense_dropout_rate),
            nn.BatchNorm1d(n_out),
            nn.SELU(),
        )

    def forward(self, x):
        # print(x.size())
        x = self.feature_extraction_layer(x)
        # global max pool 1D
        print(x.size())

        # x = self.max_l(x)
        # print(x.size())
        # conv layers
        # x = torch.max(x, dim=-1).values
        for layer in self.conv_layers:
            x = layer(x)
        # x = self.feature_abstraction_layer(x)
        # global max pool 1D
        # x = self.max_l(x)
        # x = torch.max(x, dim=-1).values
        # print(x.size())
        # x = x.view(x.size(0), -1)  # Flattening the tensor to [batch_size, features]
        # ic(x.shape)
        # fully connected layers
        
        # print('--',x.size())
        
        for layer in self.dense_layers:
            x = layer(x)
        # x = self.dense_layers(x)
        ic(x.shape)
        x = self.prediction_layer(x)
        ic(x.shape)
        return x
# model = Original_Model(
# num_classes=5,
# num_filters=64,
# num_conv_layers=2,
# num_dense_neurons=50, # batch_size = 64
# num_dense_layers=3,
# return_logits=True,
# conv_dropout_rate=0.0,
# dense_dropout_rate=0.2
# ).to(device)

# epoch = 250
# batch_size = 256
# # lr = 0.0085
# # lr = 0.00002
# lr = 0.001

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8)
# test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
# # criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
# # criterion = F.cross_entropy
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)

# torch.cuda.empty_cache()
# import gc; gc.collect()
# # ic.enable()
# ic.disable()

# train_epoch_loss = []
# test_epoch_loss = []

# print(f'./graphs1/loss_lr_{lr}_SINGLE.png')


# for e in tqdm(range(1, epoch+1)):
#     model.train()
#     train_batch_loss = []
#     test_batch_loss = []
    
#     for x_train, y_train in train_loader:
#         x_batch = torch.squeeze(x_train, 0).to(device)
#         y_batch = y_train.to(device)
#         x_batch = x_batch.float()
#         # y_batch = y_batch.view(-1)

#         # y_batch = one_hot_torch(y).to(device)
#         # print('batch y size before flatten:',y_batch.size())
#         # y_batch = y_batch.flatten()
#         # print('batch y size after flatten:',y_batch.size())
#         # print(x_batch.size())
#         # print(x_batch.size())
#     # For example, if you have a convolutional layer with 64 output channels, 3 input channels, and a kernel size of 3x3, the weight parameters would have a dimension of (64, 3, 3, 3)
#         # print(x_batch.size())
#         pred = model(x_batch.float())
#         # print(x_batch)

#         pred = pred.float()
#         # pred = pred.unsqueeze(0)
#         # ic(pred)
#         # ic(y_batch)
#         loss_train = criterion(pred, y_batch)
#         train_batch_loss.append(loss_train)
        
#         optimizer.zero_grad()
#         loss_train.backward()
#         optimizer.step()
#         # print(f'Batch - GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')

#         train_epoch_loss.append(torch.mean(torch.stack(train_batch_loss)).detach().cpu().numpy())
#     model.eval()
#     with torch.no_grad():
#         # print('test')
#         for x_test, y_test in test_loader:
#             x_batch = x_test.to(device)
#             y_batch = y_test.to(device)
            
#             # print(x_batch.size())
#             # y_batch = torch.Tensor.float(y).to(device)
#             # x_batch = x_batch.permute(0, 3, 1, 2).to(device)
#             pred = model(x_batch.float())
#             # pred = pred.unsqueeze(0)
#             # loss_test = criterion(y_batch, pred)
            
#             loss_test = criterion(pred, y_batch)
#             test_batch_loss.append(loss_test)
#         test_epoch_loss.append(torch.mean(torch.stack(test_batch_loss)).detach().cpu().numpy())
            
#     print(f'Epoch {e}')
#     print(f"Training loss: {torch.mean(torch.stack(train_batch_loss)).detach().cpu().numpy()}")
#     print(f"Validation loss: {torch.mean(torch.stack(test_batch_loss)).detach().cpu().numpy()}") 
    # scheduler.step(torch.mean(torch.stack(test_batch_loss)))
    # print(train_batch_loss)
    # print(test_batch_loss)
    # print(f"Training loss: {np.mean(train_batch_loss)}")
    # print(f"Validation loss: {np.mean(test_batch_loss)}")
# print('==='*10)
# torch.save(model.state_dict(), '/mnt/storageG1/lwang/Projects/tb_dr_MIC/saved_models/origional_model_simple.pt')


# fig, ax = plt.subplots()
# x = np.arange(1, len(train_epoch_loss)+1, 1)
# ax.plot(x, train_epoch_loss,label='Training')
# # ax.plot(x, test_epoch_loss,label='Validation')
# ax.legend()
# ax.set_xlabel("Number of Epoch")
# ax.set_ylabel("Loss")
# ax.set_xticks(np.arange(0, epoch+1, 10))
# ax.set_title(f'Loss: Learning_rate:{lr}')
# # ax_2 = ax.twinx()
# # ax_2.plot(history["lr"], "k--", lw=1)
# # ax_2.set_yscale("log")
# # ax.set_ylim(ax.get_ylim()[0], history["training_losses"][0])
# ax.grid(axis="x")
# fig.tight_layout()
# fig.show()
# fig.savefig(f'./graphs1/loss_lr_{lr}_long.png')
# print(f'./graphs1/loss_lr_{lr}.png')

#%%
torch.cuda.empty_cache()

class Model(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_classes=1,
        num_filters=64,
        filter_length=25,
        num_conv_layers=3,
        filter_scaling_factor=1.5,  # New parameter
        num_dense_neurons=256,
        num_dense_layers=2,
        conv_dropout_rate=0.0,
        dense_dropout_rate=0.0,
        return_logits=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.conv_dropout_rate = conv_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.return_logits = return_logits
        
        # now define the actual model
        self.feature_extraction_layer = self._conv_layer(
            in_channels, num_filters, filter_length
        )
        
        #dynamic filter scaling from deepram
        current_num_filters = num_filters
        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layers):
            layer = self._conv_layer(current_num_filters, int(current_num_filters * filter_scaling_factor), 3)
            self.conv_layers.append(layer)
            current_num_filters = int(current_num_filters * filter_scaling_factor)

        self.dense_layers = nn.ModuleList(
            self._dense_layer(input_dim, num_dense_neurons)
            for input_dim in [current_num_filters]
            + [num_dense_neurons] * (num_dense_layers - 1) #how does this work?
        )
        self.prediction_layer = (
            nn.Linear(num_dense_neurons, num_classes)
            if return_logits
            else nn.Sequential(nn.Linear(num_dense_neurons, num_classes), nn.ReLU()) #difference between sequential and nn.moduleList?
        )
        
        self.apply(self.init_weights)    
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _conv_layer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Dropout(p=self.conv_dropout_rate),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def _dense_layer(self, n_in, n_out):
        return nn.Sequential(
            nn.Dropout(p=self.dense_dropout_rate),
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out),
            nn.ReLU(),
        )

    def forward(self, x):
        # first pass over input
        # print(x.size())
        x = self.feature_extraction_layer(x)
        # conv layers
        for layer in self.conv_layers:
            x = layer(x)
        # global max pool 1D
        x = torch.max(x, dim=-1).values
        x = x.view(x.size(0), -1)  # Flattening the tensor to [batch_size, features]
        # ic(x.shape)
        # fully connected layers
        for layer in self.dense_layers:
            x = layer(x)
        ic(x.shape)
        x = self.prediction_layer(x)
        ic(x.shape)
        return x

# model = Original_Model(
# num_classes=5,
# num_filters=64,
# num_conv_layers=3,
# num_dense_neurons=50, # batch_size = 64
# num_dense_layers=3,
# return_logits=True,
# conv_dropout_rate=0.0,
# dense_dropout_rate=0.0
# ).to(device)

# epoch = 350
# batch_size = 64
# # lr = 0.0085
# # lr = 0.00002
# lr = 0.001

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True ,num_workers=8)
# test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
# # criterion = nn.MSELoss()
# criterion = masked_MSE
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)

def save_to_file(file_path, appendix, epoch, lr, dr, train_loss, test_loss):
    train_loss = [float(arr) for arr in train_loss]
    test_loss = [float(arr) for arr in test_loss]
    with open(file_path, "a") as f:
        f.write(f"#>> {appendix}, Epoch: {epoch}, LR: {lr}, DR: {dr}\n")
        f.write(f"Train_Loss= {train_loss}\n")
        f.write(f"Test_Loss= {test_loss}\n")
        f.write(f"lossGraph(Train_Loss, Test_Loss, '{appendix}-Epoch-{epoch}-LR-{lr}-DR-{dr}')\n")
        
def hyper_params_test(appendix, lr, fc_dr=0.2,cnn_dr=0, epoch=50, l2=0):
    print('lr:', lr, '| fc_dr:',  fc_dr, '| cnn_dr:', cnn_dr, '==='*10)
    torch.cuda.empty_cache()
    import gc; gc.collect()
    
    # stdout, stderr = run_bash_command('nvidia-smi')
    # print('STDOUT:', stdout)
    # print('STDERR:', stderr)
    
    model = Model(
    num_classes=3,
    num_filters=64,
    num_conv_layers=2,
    num_dense_neurons=256, # batch_size = 64
    # num_dense_neurons=128, # batch_size = 64
    num_dense_layers=2,
    return_logits=False,
    conv_dropout_rate=0.00,
    dense_dropout_rate=0.5
    ).to(device)
    
    # model = Original_Model(
    # num_classes=5,
    # num_filters=64,
    # num_conv_layers=3,
    # num_dense_neurons=64, # batch_size = 64
    # num_dense_layers=4,
    # return_logits=True,
    # dense_dropout_rate=fc_dr,
    # conv_dropout_rate=cnn_dr
    # ).to(device)
    
    # stdout, stderr = run_bash_command('nvidia-smi')
    # print('STDOUT:', stdout)
    # print('STDERR:', stderr)
    
    epoch = epoch
    batch_size = 128
    lr = lr

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)
    

    ic.disable()
    # ic.enable()
        
    train_epoch_loss = []
    test_epoch_loss = []

    # print(f'./graphs1/loss_lr_{lr}_SINGLE.png')

    for x, y in train_loader:
        x = x
        y = y
        break

    for e in tqdm(range(1, epoch+1)):
        model.train()
        train_batch_loss = []
        test_batch_loss = []
        
        # for x, y in train_loader:
        x_batch = torch.squeeze(x, 0).to(device)
        y_batch = y.to(device)
        x_batch = x_batch.float()
        # y_batch = y_batch.view(-1)

        # y_batch = one_hot_torch(y).to(device)
        # print('batch y size before flatten:',y_batch.size())
        # y_batch = y_batch.flatten()
        # print('batch y size after flatten:',y_batch.size())
        # print(x_batch.size())
        # print(x_batch.size())
    # For example, if you have a convolutional layer with 64 output channels, 3 input channels, and a kernel size of 3x3, the weight parameters would have a dimension of (64, 3, 3, 3)
        # print(x_batch.size())
        pred = model(x_batch.float())
        # print(x_batch)
        # print(pred)
        # pred = pred.unsqueeze(0)
        # ic(pred)
        # ic(y_batch)
        ic(pred.size())
        loss_train = criterion(pred, y_batch)
        ic(loss_train)
        train_batch_loss.append(loss_train)
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        # print(f'Batch - GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')

        train_epoch_loss.append(torch.mean(torch.stack(train_batch_loss)).detach().cpu().numpy())
        model.eval()
        with torch.no_grad():
            # print('test')
            for x, y in test_loader:
                x_batch = x.to(device)
                y_batch = y.to(device)
                # print(x_batch.size())
                # y_batch = torch.Tensor.float(y).to(device)
                # x_batch = x_batch.permute(0, 3, 1, 2).to(device)
                pred = model(x_batch.float())
                # pred = pred.unsqueeze(0)
                loss_test = criterion(pred, y_batch)
                test_batch_loss.append(loss_test)
            test_epoch_loss.append(torch.mean(torch.stack(test_batch_loss)).detach().cpu().numpy())
    fig, ax = plt.subplots()
    x = np.arange(1, epoch+1, 1)      
    ax.plot(x, train_epoch_loss,label='Training')
    ax.plot(x, test_epoch_loss,label='Validation')
    ax.legend()
    ax.set_xlabel("Number of Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(np.arange(0, epoch+1, 10))
    ax.set_title(f'Loss: Learning_rate:{lr}, cnn_dr:{cnn_dr}, cnn_dr:{fc_dr}')
    # ax_2 = ax.twinx()
    # ax_2.plot(history["lr"], "k--", lw=1)
    # ax_2.set_yscale("log")
# ax.set_ylim(ax.get_ylim()[0], history["training_losses"][0])
    ax.grid(axis="x")
    fig.tight_layout()
    fig.show()
    # fig.savefig(f'./graphs1/{appendix}_loss_lr_{lr}_cnn_dr_{cnn_dr}_fc_dr_{fc_dr}.png')
    save_to_file('/mnt/storageG1/lwang/Projects/TBpt/Analysis/trials.txt', appendix ,epoch, lr, fc_dr, train_epoch_loss, test_epoch_loss)
    print(f'./graphs1/{appendix}_loss_lr_{lr}_cnn_dr_{cnn_dr}_fc_dr_{fc_dr}.png')
    
#%%
epoch = 200
for lr in [0.01, 0.001, 1e-3, 1e-4, 1e-5, 1e-6]:
    # for fc_dr in [:
    for fc_dr in [0]:
        for cnn_dr in [0]:
            hyper_params_test('128-3-50', lr, fc_dr, cnn_dr, epoch=epoch)