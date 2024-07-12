import torch
import numpy as np 
device = torch.device('cuda')
import pandas as pd
from models.my_models import *
from models.models_config import get_model_config
#Different Domain Adaptation  approaches
import importlib
import random
#from trainer.cross_domain_models.ATL_NCE import cross_domain_train

print(torch.cuda.current_device())

def main():
    select_method='ATL_NCE'
    # hyper parameters
    hyper_param={'FD001_FD002': {'epochs':75,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
                  'FD001_FD003': {'epochs':75,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
                  'FD001_FD004': {'epochs':75,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001},
                  'FD002_FD001': {'epochs':20,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001},
                  'FD002_FD003': {'epochs':20,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001},
                  'FD002_FD004': {'epochs':20,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
                  'FD003_FD001': {'epochs':100,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
                  'FD003_FD002': {'epochs':100,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
                  'FD003_FD004': {'epochs':100,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
                  'FD004_FD001': {'epochs':150,'batch_size':256,'lr':3e-4,'d_lr':3e-4, 'alpha_nce':0.2},
                  'FD004_FD002': {'epochs':150,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
                  'FD004_FD003': {'epochs':150,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001}}
    # load dataset
    data_path= "./data/processed_data/cmapps_train_test_cross_domain.pt"
    my_dataset = torch.load(data_path)

    # configuration setup
    config = get_model_config('CNN_AE')
    config.update({'num_runs':1, 'save':True, 'tensorboard':False,'tsne':False,'tensorboard_epoch':False,'k_disc':100, 'k_clf':1,'iterations':1})

    if config['tensorboard']:
      wandb.init(project="Domain Adaptation for with Contrastive Coding",name=f"{select_method}",dir= "/home/emad/Mohamed2/ATL_NCE/visualize/", sync_tensorboard=True)
      wandb.config  = hyper_param



if __name__ == '__main__':
  select_method= 'CT_AE' #'T_AE': target auto encoder+discrim; 'T_DA': target discrim; 'ATL_NCE': nce loss (bug exists when adapt to CNN)
  method = importlib.import_module(f'trainer.cross_domain_models.{select_method}')
  # hyper parameters
#  hyper_param={ 'FD001_FD002': {'epochs':50,'batch_size':256,'lr':5e-4,'d_lr':5e-4, 'alpha_nce':0.2},
#                  'FD001_FD003': {'epochs':30,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
#                  'FD001_FD004': {'epochs':40,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001},
#                  'FD002_FD001': {'epochs':50,'batch_size':256,'lr':5e-4,'d_lr':5e-4, 'alpha_nce':0.001},
#                  'FD002_FD003': {'epochs':50,'batch_size':256,'lr':5e-4,'d_lr':5e-4, 'alpha_nce':0.001},
#                  'FD002_FD004': {'epochs':50,'batch_size':256,'lr':5e-4,'d_lr':5e-4, 'alpha_nce':0.2},
#                  'FD003_FD001': {'epochs':50,'batch_size':256,'lr':5e-4,'d_lr':5e-4, 'alpha_nce':0.2},
#                  'FD003_FD002': {'epochs':50,'batch_size':256,'lr':5e-4,'d_lr':5e-4, 'alpha_nce':0.2},
#                  'FD003_FD004': {'epochs':50,'batch_size':256,'lr':5e-4,'d_lr':5e-4, 'alpha_nce':0.2},
#                  'FD004_FD001': {'epochs':50,'batch_size':256,'lr':5e-4,'d_lr':5e-4, 'alpha_nce':0.2},
#                  'FD004_FD002': {'epochs':50,'batch_size':256,'lr':5e-4,'d_lr':5e-4, 'alpha_nce':0.2},
#                  'FD004_FD003': {'epochs':50,'batch_size':256,'lr':5e-4,'d_lr':5e-4, 'alpha_nce':0.001}}
#  alpha: 0.01
#  hyper_param={'FD001_FD002': {'epochs':50,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
#                'FD001_FD003': {'epochs':50,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
#                'FD001_FD004': {'epochs':100,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001},
#                'FD002_FD001': {'epochs':50,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001},
#                'FD002_FD003': {'epochs':100,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001},
#                'FD002_FD004': {'epochs':100,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
#                'FD003_FD001': {'epochs':150,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
#                'FD003_FD002': {'epochs':150,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
#                'FD003_FD004': {'epochs':100,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
#                'FD004_FD001': {'epochs':150,'batch_size':256,'lr':3e-4,'d_lr':3e-4, 'alpha_nce':0.2},
#                'FD004_FD002': {'epochs':200,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
#                'FD004_FD003': {'epochs':20,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001}}
  hyper_param={'FD001_FD002': {'epochs':200,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
              'FD001_FD003': {'epochs':250,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
              'FD001_FD004': {'epochs':80,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001},
              'FD002_FD001': {'epochs':40,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001},
              'FD002_FD003': {'epochs':140,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001},
              'FD002_FD004': {'epochs':130,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},  
              'FD003_FD001': {'epochs':140,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},  
              'FD003_FD002': {'epochs':200,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
              'FD003_FD004': {'epochs':250,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
              'FD004_FD001': {'epochs':200,'batch_size':256,'lr':3e-4,'d_lr':3e-4, 'alpha_nce':0.2},
              'FD004_FD002': {'epochs':300,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.2},
              'FD004_FD003': {'epochs':20,'batch_size':256,'lr':5e-5,'d_lr':5e-5, 'alpha_nce':0.001}}
  # load dataset
  data_path= "./data/processed_data/cmapps_train_test_cross_domain.pt"
  my_dataset = torch.load(data_path)

  # configuration setup
  config = get_model_config('LSTM') #CNN_AE
  config.update({'num_runs':1, 'save':True, 'tensorboard':False,'tsne':False,'tensorboard_epoch':False,'k_disc':100, 'k_clf':1,'iterations':1, 'permute':False, 'alpha':0.1})


  df=pd.DataFrame();res = [];full_res = []
  print('=' * 89)
  print (f'Domain Adaptation using: {select_method}')
  print('=' * 89)
  for src_id in ['FD001', 'FD002', 'FD003', 'FD004']: #'FD001', 'FD002', 'FD003', 'FD004'
      for tgt_id in ['FD001', 'FD002', 'FD003', 'FD004']: #'FD001', 'FD002', 'FD003', 'FD004'
          if src_id != tgt_id:
              total_loss = []
              total_score = []
              total_src_tgt_kl = []
              total_best_tgt_kl = []
              total_best_tgt_kl2 = []
              seed = 42
              for run_id in range(config['num_runs']):
                  seed += 1
                  torch.manual_seed(seed)
                  np.random.seed(seed)
                  random.seed(seed)
                  
                  src_only_loss, src_only_score, test_loss, test_score, src_tgt_kl_loss, best_tgt_kl_loss, best_tgt_kl_loss2 = method.cross_domain_train(hyper_param,device,config,LSTM_RUL,my_dataset,src_id,tgt_id,run_id) #CNN_RUL
                  total_loss.append(test_loss)
                  total_score.append(test_score)
                  total_src_tgt_kl.append(src_tgt_kl_loss)
                  total_best_tgt_kl.append(best_tgt_kl_loss)
                  total_best_tgt_kl2.append(best_tgt_kl_loss2)
              loss_mean, loss_std = np.mean(np.array(total_loss)), np.std(np.array(total_loss))
              score_mean, score_std = np.mean(np.array(total_score)), np.std(np.array(total_score))
              src_tgt_kl_mean, src_tgt_kl_std = np.mean(np.array(total_src_tgt_kl)), np.std(np.array(total_src_tgt_kl))
              best_tgt_kl_mean, best_tgt_kl_std = np.mean(np.array(total_best_tgt_kl)), np.std(np.array(total_best_tgt_kl))
              best_tgt_kl_mean2, best_tgt_kl_std2 = np.mean(np.array(total_best_tgt_kl2)), np.std(np.array(total_best_tgt_kl2))
              full_res.append((f'run_id:{run_id}',f'{src_id}-->{tgt_id}', f'{src_only_loss:2.4f}' ,f'{loss_mean:2.4f}',f'{loss_std:2.4f}',f'{src_only_score:2.4f}',f'{score_mean:2.4f}',f'{score_std:2.4f}',f'{src_tgt_kl_mean:2.4f}',f'{best_tgt_kl_mean:2.4f}',f'{best_tgt_kl_mean2:2.4f}'))
              
  df= df.append(pd.Series((f'{select_method}')), ignore_index=True)
  df= df.append(pd.Series(("run_id", 'scenario','src_only_loss', 'mean_loss','std_loss', 'src_only_score', f'mean_score',f'std_score', 'kl(src, best_tgt)', 'kl(best_tgt, DA)', 'kl(DA, best_tgt)')), ignore_index=True)
  df = df.append(pd.DataFrame(full_res), ignore_index=True)
  print('=' * 89)
  print (f'Results using: {select_method}')
  print('=' * 89)
  print(df.to_string())
  df.to_csv(f'./results/Final_results_{select_method}_{config["alpha"]}_dtw.csv')
