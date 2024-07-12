import sys

sys.path.append("..")
from utils import *
from data.mydataset import create_dataset_full, create_dataset_cluster
import torch
from torch import nn
import matplotlib.pyplot as plt
from trainer.train_eval import evaluate
import copy
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from trainer.cross_domain_models.NCE_model import dicriminator
from models.my_models import LSTM_decoder
from tslearn.clustering import TimeSeriesKMeans


def cross_domain_train(params,device, config, model, my_dataset, src_id, tgt_id, run_id):
    hyper = params[f'{src_id}_{tgt_id}']
    print(f'From_source:{src_id}--->target:{tgt_id}...')
    
    print('Restore source pre_trained model...')
    checkpoint = torch.load(f'./trained_models/single_domain2/pretrained_{config["model_name"]}_{src_id}_new.pt')
    tgt_checkpoint = torch.load(f'./trained_models/single_domain2/pretrained_{config["model_name"]}_{tgt_id}_new.pt')
    da_tgt_checkpoint = torch.load(f'./trained_models/cross_domains/{src_id}_to_{tgt_id}_{run_id}_{config["alpha_dtw"]}_tgt.pt')
    tgt_de_checkpoint = torch.load(f'./trained_models/cross_domains/{src_id}_to_{tgt_id}_{run_id}_{config["alpha_dtw"]}_tgt_de.pt')
    dis_checkpoint = torch.load(f'./trained_models/cross_domains/{src_id}_to_{tgt_id}_{run_id}_{config["alpha_dtw"]}_dis.pt')

    source_model = model(14, 32, 5, 0.5, True, device).to(device)

    print('=' * 89)
    print(f'The {config["model_name"]} has {count_parameters(source_model):,} trainable parameters')
    print('=' * 89)
    source_model.load_state_dict(checkpoint['state_dict'])
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    source_encoder = source_model.encoder


    # initialize target model
    #target_model = model(14, 32, 0.5).to(device)
    target_model = model(14, 32, 5, 0.5, True, device).to(device)
    target_model.load_state_dict(da_tgt_checkpoint)
    target_encoder = target_model.encoder

    target_decoder = LSTM_decoder(14, 32, 5, 0.5, True, device).to(device)
    target_decoder.load_state_dict(tgt_de_checkpoint)
    target_encoder.train()
    target_decoder.train()
    
    #best_target_model = model(14, 32, 0.5).to(device)
    best_target_model = model(14, 32, 5, 0.5, True, device).to(device)
    best_target_model.load_state_dict(tgt_checkpoint['state_dict'])
    # discriminator network
    discriminator = dicriminator().to(device)
    discriminator.load_state_dict(dis_checkpoint)
    
    # criterion
    criterion = RMSELoss()
    criterion_dtw = SoftDTW(gamma=0.01, normalize=True)
    dis_critierion = nn.BCEWithLogitsLoss()
    # optimizer
    discriminator_optim = torch.optim.AdamW(discriminator.parameters(), lr=hyper['d_lr'], betas=(0.5, 0.9))
    target_optim = torch.optim.AdamW([{'params': target_encoder.parameters()}, {'params': target_decoder.parameters(), 'lr': 100*hyper['lr']}], lr=hyper['lr'], betas=(0.5, 0.9))


    print('Clustering target data...')
    src_trainX, _, src_trainY, _ = my_dataset[src_id]
    src_trainY = np.array(src_trainY)
    
    tgt_trainX, _, _, _ = my_dataset[tgt_id]
    
    src_trainY_cluster = src_trainY.copy()
    src_trainY_cluster[src_trainY <= 0.15] = 0 
    src_trainY_cluster[src_trainY > 0.66] = 2
    src_trainY_cluster[(src_trainY > 0.15) & (src_trainY <= 0.66 )] = 1    


    km_sdtw = TimeSeriesKMeans(n_clusters=config['n_cluster'], metric="softdtw", max_iter=5, random_state=0).fit(tgt_trainX)  #euclidean, dtw, softdtw
    y_pred_km_sdtw = km_sdtw.predict(tgt_trainX)
    print('kmeans soft dtw y_pred', np.unique(y_pred_km_sdtw, return_counts=True))
    print('trainY_cluster', np.unique(src_trainY_cluster, return_counts=True))
    
    clustering_entrogy = [0] * config['n_cluster']
    true_entrogy = [0] * config['n_cluster']
    for i in range(tgt_trainX.shape[2]):
      for i_clu in range(config['n_cluster']):
        clustering_entrogy[i_clu] += np.var(tgt_trainX[y_pred_km_sdtw==i_clu, :, i])

    tgt_clustering_idx = np.argsort(clustering_entrogy)

    print('sdtw', clustering_entrogy)
    print('true', true_entrogy)
    print(tgt_clustering_idx)

    for epoch in range(1, hyper['epochs'] + 1):
        for cluster_idx in range(3):
          src_train_dl, src_test_dl = create_dataset_cluster(my_dataset[src_id], src_trainY_cluster==cluster_idx, batch_size=hyper['batch_size'])
          tgt_train_dl, tgt_test_dl = create_dataset_cluster(my_dataset[tgt_id], y_pred_km_sdtw==tgt_clustering_idx[2-cluster_idx], batch_size=hyper['batch_size'])
        
          batch_iterator = zip(loop_iterable(src_train_dl), loop_iterable(tgt_train_dl))
          total_loss = 0
          total_accuracy = 0

          target_losses, recon_losses = 0, 0
          start_time = time.time()
          for _ in range(config['iterations']):  # , leave=False):
              # Train discriminator
              set_requires_grad(target_encoder, requires_grad=False)
              set_requires_grad(target_decoder, requires_grad=False)
              set_requires_grad(discriminator, requires_grad=True)
              for _ in range(config['k_disc']):
                  (source_x, _), (target_x, _) = next(batch_iterator)
                  if config['permute']==True:
                      source_x = source_x.permute(0, 2, 1)
                      target_x = target_x.permute(0, 2, 1)
                  source_x, target_x = source_x.to(device), target_x.to(device)
                  _, source_features, _ = source_model(source_x) 
                  _, target_features, _ = target_model(target_x)
  
                  discriminator_x = torch.cat([source_features, target_features]) #.view(target_features.shape[0], -1)
                  discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                               torch.zeros(target_x.shape[0], device=device)])
  
                  preds = discriminator(discriminator_x).squeeze()
                  loss = dis_critierion(preds, discriminator_y)
                  discriminator_optim.zero_grad()
                  loss.backward()
                  discriminator_optim.step()
                  total_loss += loss.item()
                  total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()
              # Train Feature Extractor
              set_requires_grad(target_encoder, requires_grad=True)
              set_requires_grad(target_decoder, requires_grad=True)
              set_requires_grad(discriminator, requires_grad=False)
              for _ in range(config['k_clf']):
                  target_optim.zero_grad()
                  # Get a batch
                  _, (target_x, _) = next(batch_iterator)
                  if config['permute']==True:
                      target_x = target_x.permute(0, 2, 1)
                  target_x = target_x.to(device)              
                  _, target_features, encoder_outputs = target_model(target_x)
                  target_reconstruction = target_decoder(encoder_outputs)
                  
                  target_recon_loss = criterion_dtw(target_x, target_reconstruction)
                  target_recon_loss = torch.mean(target_recon_loss)
                  # flipped labels
                  discriminator_y = torch.ones(target_x.shape[0], device=device)
                  preds = discriminator(target_features).squeeze() #.view(target_features.shape[0], -1)
                  target_loss = dis_critierion(preds, discriminator_y)
  
                  #total loss
                  loss = target_loss + config['alpha_dtw']*target_recon_loss 
                  loss.backward()
                  target_optim.step()
                  target_losses += target_loss.item()
                  recon_losses += target_recon_loss.item()
                  
          mean_loss = total_loss / (config['iterations'] * config['k_disc'])
          mean_accuracy = total_accuracy / (config['iterations'] * config['k_disc'])
          mean_recog_loss = recon_losses / (config['iterations'] * config['k_clf'])
          mean_tgt_loss = target_losses / (config['iterations'] * config['k_clf'])
  
          # tensorboard logging
          # log time
          end_time = time.time()
          epoch_mins, epoch_secs = epoch_time(start_time, end_time)
          print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
          print(f'Discriminator_loss:{mean_loss} \t Discriminator_accuracy{mean_accuracy}')
          print(f'target_loss:{mean_tgt_loss}  \t target_recon_loss{mean_recog_loss}')
          if epoch % 1 == 0:
              src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, config,device)
              test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, config,device)
              print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
              print(f'DA RMSE:{test_loss} \t DA Score:{test_score}')
    

    src_only_loss, src_only_score, src_only_fea, _, pred_labels, true_labels = evaluate(source_model, tgt_test_dl, criterion, config, device)
    test_loss, test_score, target_fea, _, pred_labels_DA, true_labels_DA = evaluate(target_model, tgt_test_dl, criterion, config, device)
    _, _, best_target_fea, _, _, _ = evaluate(best_target_model, tgt_test_dl, criterion, config, device)
    kl_critierion = nn.KLDivLoss(reduction="batchmean", log_target=False)
    src_tgt_kl_loss = kl_critierion(torch.log_softmax(src_only_fea, dim=1), torch.softmax(best_target_fea, dim=1))
    best_tgt_kl_loss = kl_critierion(torch.log_softmax(best_target_fea, dim=1), torch.softmax(target_fea, dim=1))
    best_tgt_kl_loss2 = kl_critierion(torch.log_softmax(target_fea, dim=1), torch.softmax(best_target_fea, dim=1))

    
    print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
    print(f'After DA RMSE:{test_loss} \t After DA Score:{test_score}')
    print(f'KL(src_only, best_target):{src_tgt_kl_loss} \t KL(best_target, DA):{best_tgt_kl_loss} \t KL(DA, best_target):{best_tgt_kl_loss2}')
    # print the true and predicted labels
    if config['tensorboard']:
        if config['tsne']:
            _, _, src_features, _, _, _ = evaluate(source_model, src_train_dl, criterion, config,device)
            _, _, tgt_features, _, _, _ = evaluate(source_model, tgt_train_dl, criterion, config,device)
            _, _, tgt_trained_features, _, _, _ = evaluate(target_model, tgt_train_dl, criterion, config,device)

    if config['save']:
        torch.save(target_model.state_dict(), f'./trained_models/cross_domains/{src_id}_to_{tgt_id}_{run_id}_{config["alpha_dtw"]}_new_66.pt')
    return src_only_loss, src_only_score, test_loss, test_score, src_tgt_kl_loss.item(), best_tgt_kl_loss.item(), best_tgt_kl_loss2.item()
