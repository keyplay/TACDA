##############################################################################
# All the codes about the model construction should be kept in the folder ./models/
# All the codes about the data processing should be kept in the folder ./data/
# All the codes about the loss functions should be kept in the folder ./losses/
# All the source pre-trained checkpoints should be kept in the folder ./trained_models/
# All runs and experiment
# The file ./trainer/ stores the training and test strategy
# The file ./main.py should be simple
#################################################################################
# imports
import time

import torch
from torch import nn
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
import pandas as pd
import argparse
from tqdm import tqdm, trange
import warnings
from utils import *
from models.my_models import *
from trainer.train_eval import train, evaluate
from trainer.pre_train_test_split import pre_train
from sklearn.exceptions import DataConversionWarning
from models.models_config import get_model_config, initlize
from data.mydataset import create_dataset, create_dataset_full
# torch.nn.Module.dump_patches = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Steps
# Step 1: Load Dataset
# Step 2: Create Model Class
# Step 3: Load Model
# Step 2: Make Dataset Iterable
# Step 4: Instantiate Model Class
# Step 5: Instantiate Loss Class
# Step 6: Instantiate Optimizer Class
# Step 7: Train Mode

"""Configureations"""
params = {'window_length': 30, 'sequence_length': 30, 'batch_size': 256, 'input_dim': 14, 'src_pretrain': False, 'pretrain_epoch': 30, 'save': False,
          'data_path': r"./data/processed_data/cmapps_train_test_cross_domain.pt", "data_type": 'log',
          'dropout': 0.5,  'lr': 1e-3, 'tensorboard':False}


# load data
my_dataset = torch.load(params['data_path'])
# load model
config = get_model_config('LSTM') #CNN_AE



def main():
    df=pd.DataFrame()
    res = []
    # pm = Symbol(u'±')
    full_res = []
    for src_id in ['FD001', 'FD002', 'FD003','FD004']:
        print('Initializing model for', src_id)
        #source_model = CNN_RUL(params['input_dim'], 32, 0.5).to(device)
        source_model = LSTM_RUL(14, 32, 5, 0.5, True, device).to(device)
        #source_model.apply(weights_init)
        print('=' * 89)
        print(f'The {config["model_name"]} has {count_parameters(source_model):,} trainable parameters')
        print('=' * 89)
        print('Load_source_target datasets...')
        src_train_dl, src_test_dl = create_dataset_full(my_dataset[src_id], batch_size=params['batch_size'])
    
        trained_source_model = pre_train(source_model, src_train_dl, src_test_dl, src_id, config, params)
        
        # test on cross domain data
        for tgt_id in ['FD001', 'FD002', 'FD003', 'FD004']:
            if src_id != tgt_id:
                tgt_train_dl, tgt_test_dl = create_dataset_full(my_dataset[tgt_id])
                criterion = RMSELoss()
                test_loss, test_score, _, _, _, _ = evaluate(trained_source_model, tgt_test_dl, criterion, config, device)
                print('=' * 89)
                print(f'\t  Performance on test set:{tgt_id}::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')
                print('=' * 89)  


main()
print('Finished')
print('Finished')
