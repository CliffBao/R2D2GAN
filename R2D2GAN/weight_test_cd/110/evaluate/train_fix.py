from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest

from model_gan import highwayNet_g_compose
from utils_fix import ngsimDatasetGan, composedDataset
from torch.utils.data import DataLoader
import scipy.io as sio
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os
plt.switch_backend('agg')

# when a new gan model is trained
# copy train,model,util files here and add a gan postfix
# ngsimDataset in utils_gan also add Gan postfix
# generator arguments copied from train_gan and ignore args_d
# check generator input arguments and dataloader output numbers
# copy raw training and testing data from data/raw to data folder
args_d = {}
args_d['use_cuda'] = True
args_d['in_length'] = 81
args_d['out_length'] = 81
args_d['encoder_size'] = 64
args_d['input_embedding_size_d'] = 32
args_d['batch_size'] = 64
args_d['traj_num'] = 4

args_g={}
args_g['use_cuda'] = True
args_g['out_length'] = 81
args_g['input_embedding_size_g'] = 64
args_g['encoder_size'] = 128
args_g['batch_size'] = args_d['batch_size']
args_g['latent_dim'] = 16
args_g['traj_num'] = 4

# args_gd = {}
# args_gd['use_cuda'] = True
# args_gd['in_length'] = 81
# args_gd['out_length'] = 81
# args_gd['encoder_size'] = 512
# args_gd['latent_dim'] = 100
# args_gd['batch_size'] = 128
# args_gd['input_embedding_size_g'] = 128
#
# args_vae = {}
# args_vae['in_length'] = 81*2
# args_vae['input_embedding_size'] = 512
# args_vae['encoder_size'] = 256
# args_vae['z_dim'] = 128
# args_vae['traj_num'] = 4

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 256
args['decoder_size'] = 256
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 64
args['input_embedding_size'] = 64
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = False
args['train_flag'] = True

# 这里需要注意scale需要以前30步为标准建立，如果以80步建立就会得到很大的偏差
# Initialize network
net = highwayNet(args)
net_g_compose = highwayNet_g_compose(args_g)
# net_g_decompose = highwayNet_g_decompose(args_gd)
# net_vae = VAE(args_vae)

# net_g_decompose.load_state_dict(torch.load('trained_models/ngsim_wcgenerator_decompose_us1011.tar'))
net_g_compose.load_state_dict(torch.load('trained_models/ngsim_generator_decompose_us1011_nei3_epoch1389.tar'))
# net_vae.load_state_dict(torch.load('trained_models/ngsim_vae_decompose_us1011_nei3_epoch483.tar'))
if args['use_cuda']:
    net = net.cuda()
    # net_vae = net_vae.cuda()
    net_g_compose = net_g_compose.cuda()
    # net_g_decompose = net_g_decompose.cuda()

## Initialize optimizer
pretrainEpochs = 15
trainEpochs = 6
optimizer = torch.optim.Adam(net.parameters())
batch_size = args_g['batch_size']
crossEnt = torch.nn.BCELoss()
t_h = 30
t_f = 50
d_s = 2

## Initialize data loaders
real_set_name = 'data/TrainSet-us1011-gan-smooth-nei3.mat'
composed_set_name = 'data/gen/composed_us1011_nei3.mat'
# real_set_name = 'data/TrainSet-carla-1-nei1-maxgrid.mat'
hero_grid_seq = 19
t_short = args_g['out_length']
realSet = ngsimDatasetGan(mat_file=real_set_name, enc_size=256,neigh_num=args_g['traj_num']-1,
                        t_f=args_g['out_length'],
                        batch_size=args_g['batch_size'], length = args_g['out_length'],
                        local_normalise=1)
all_index = realSet.get_all_index()
all_index = all_index.cuda()
composed_dataset = torch.zeros(all_index.shape[0],args_d['traj_num']*2*args_d['in_length'])
div_num = 30
with torch.no_grad():
    for i in range(div_num):
        print('generating part ',i,'/',div_num)
        start_id = all_index.shape[0]//div_num*i
        end_id = all_index.shape[0]//div_num*(i+1)
        composed_dataset[start_id:end_id,:] = net_g_compose(all_index[start_id:end_id,:])
        torch.cuda.empty_cache()
        if i == div_num-1:
            composed_dataset[end_id:, :] = net_g_compose(all_index[end_id:, :])


sio.savemat(composed_set_name,{'data':composed_dataset.detach().cpu().numpy(),
                               'label':all_index.detach().cpu().numpy()})

trSet = composedDataset(mat_file=composed_set_name)
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=4,
                          drop_last=True,collate_fn=trSet.collate_fn)

enc_size = args['encoder_size']
valSet = ngsimDataset('data/ValSet-us1011-gan-smooth-nei3.mat',grid_size=args['grid_size'],
                      t_h = t_h,t_f = t_f,d_s = d_s,enc_size=enc_size)
#valSet = ngsimDataset('data/ValSet-carla-9-nei1-maxgrid.mat',grid_size=args['grid_size'],
#                      t_h = t_h,t_f = t_f,d_s = d_s,enc_size=enc_size)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=valSet.collate_fn)

def scene2regdata(scene,hero_index,nbr_index,index_division):
    t_h = 30
    t_f = 50
    d_s = 2
    scene = scene.view(-1,2,args_g['out_length'])
    hero_traj = scene[hero_index,:,:]
    ref_pose = hero_traj[:, :, t_h]

    scene = scene.view(args_g['batch_size'],args_g['traj_num'],2,args_g['out_length'])
    scene = scene-ref_pose.unsqueeze(1).unsqueeze(3)
    scene_scaled = scene
    scene_scale = abs(scene[:,:,:,0:t_h+1:d_s]).max(3,keepdim=True)[0].max(1,keepdim=True)[0]
    scene_scaled = scene/scene_scale

    scene_scaled = scene_scaled.view(-1,2,args_g['out_length'])
    hero_traj = scene_scaled[hero_index, :, :]
    nbr_traj = scene_scaled[nbr_index, :, :]

    # index_division = index_division.detach()
    for id, idx in enumerate(index_division):
        idx = np.array(idx) - id
        idx = np.arange(idx[0], idx[-1])
        index_division[id] = idx.tolist()
        # nbr_traj[idx, :, :] = nbr_traj[idx, :, :] - ref_pose[id, :].unsqueeze(1)
        # hero_traj[id, :, :] = hero_traj[id, :, :] - ref_pose[id, :].unsqueeze(1)
        # this is for generated
        # current_scene = torch.cat((nbr_traj[idx, :, :], hero_traj[id, :, :].unsqueeze(0)), 0)
        # fut_scale = abs(current_scene[:, :, 0:t_h + 1:d_s]).max(2)[0].max(0)[0]
        # nbr_traj[idx, :, :] = nbr_traj[idx, :, :] / fut_scale.unsqueeze(0).unsqueeze(2)
        # hero_traj[id, :, :] = hero_traj[id, :, :] / fut_scale.unsqueeze(0).unsqueeze(2)
    # print(nbr_traj.shape[0]-composed_mask[:,:,:,0].nonzero().shape[0])
    # index_division.requires_grad_(True)
    hist = hero_traj[:, :, 0:t_h + 1:d_s].permute(2, 0, 1)
    fut = hero_traj[:, :, t_h + d_s:t_h + t_f + 1:d_s].permute(2, 0, 1)
    nbrs = nbr_traj[:, :, 0:t_h + 1:d_s].permute(2, 0, 1)
    return hist,fut,nbrs


## Variables holding train and validation loss values:
train_loss = []
best_val_loss = 99999.9
best_tr_loss = 99999.9
val_loss = []
prev_val_loss = math.inf
draw_num = 100
draw_count = 0
for epoch_num in range(pretrainEpochs):
    if epoch_num == 0:
        print('Pre-training with MSE loss')
    elif epoch_num == pretrainEpochs:
        print('Training with NLL loss')


    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = True

    # Variables to track training performance:
    avg_tr_loss = 0
    total_tr_loss = 0
    avg_tr_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0

    for i, data in enumerate(trDataloader):

        st_time = time.time()
        composed_scene,index_division,pos_condition = data

        if args_g['use_cuda']:
            composed_scene = composed_scene.cuda()
            pos_condition = pos_condition.view(-1,2)
            hero_pos_index = torch.where(abs(pos_condition).sum(dim=1) == 0)[0]
            nbrs_pos_index = torch.where(abs(pos_condition).sum(dim=1) > 0)[0]
        hist, fut, nbrs = scene2regdata(composed_scene.detach(), hero_pos_index, nbrs_pos_index, index_division)

        # plot
        chose_one = np.random.randint(0,args_g['batch_size'])
        if draw_count<draw_num:
            plt.plot(hist[:, chose_one, 0].cpu().numpy(), hist[:, chose_one, 1].cpu().numpy(), color='r')
            plt.plot(fut[:, chose_one, 0].cpu().numpy(), fut[:, chose_one, 1].cpu().numpy(), color='b')
            neighbors = nbrs[:,index_division[chose_one],:].cpu().numpy()
            for j in range(int(neighbors.shape[1])):
                plt.plot(neighbors[:,j,0], neighbors[:,j,1],color='r')
            path = 'gen_samples/'
            fullname = np.char.add(path, str(draw_count))
            plt.savefig(fullname.item())
            plt.show()
            plt.close()
            draw_count += 1

        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            lat_enc = torch.zeros(hist.shape[1],2).cuda()
            lon_enc = torch.zeros(hist.shape[1],2).cuda()
            fut = fut.cuda()
            op_mask = torch.ones_like(fut).cuda()

        # Forward pass
        if args['use_maneuvers']:
            # print('hist size: ', hist.size())
            # print('nbrs size: ', nbrs.size())
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, lat_enc, lon_enc,index_division)
            # Pre-train with MSE loss to speed up training
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
            # Train with NLL loss
                l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            fut_pred = net(hist, nbrs, lat_enc, lon_enc,index_division)
            # print('real',fut.shape)
            # print('pred',fut_pred.shape)
            # fut = fut*scale
            # fut_pred[:,:,0:2] = fut_pred[:,:,0:2]*scale
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

        # Backprop and update weights
        # print(fut.size())
        # print(fut_pred.size())
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        # Track average train loss and average train time:
        batch_time = time.time()-st_time
        avg_tr_loss += l.item()
        total_tr_loss += l.item()
        avg_tr_time += batch_time

        if i%100 == 99:
            eta = avg_tr_time/100*(len(trSet)/batch_size-i)
            print("Epoch no:",epoch_num+1,"| Epoch progress(%):",format(i/(len(trSet)/batch_size)*100,'0.2f'), "| Avg train loss:",format(avg_tr_loss/100,'0.4f'),"| Acc:",format(avg_lat_acc,'0.4f'),format(avg_lon_acc,'0.4f'), "| Validation loss prev epoch",format(prev_val_loss,'0.4f'), "| ETA(s):",int(eta))
            train_loss.append(avg_tr_loss/100)
            avg_tr_loss = 0
            avg_lat_acc = 0
            avg_lon_acc = 0
            avg_tr_time = 0
    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    print("current loss: ", total_tr_loss, ' history best loss: ', best_tr_loss)
    if total_tr_loss < best_tr_loss or epoch_num == 0:
        best_tr_loss = total_tr_loss
        torch.save(net.state_dict(), 'trained_models/cslstm_ngsim_nei1_besttr.tar')
        print("Best tr loss updated! current best tr loss is at epoch: ", epoch_num + 1)
    sio.savemat('train_data.mat',{'real':fut.detach().cpu().numpy(),'pred':fut_pred.detach().cpu().numpy()})

    ## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = False

    print("Epoch",epoch_num+1,'complete. Calculating validation loss...')
    avg_val_loss = 0
    avg_val_lat_acc = 0
    avg_val_lon_acc = 0
    val_batch_count = 0
    total_points = 0

    for i, data in enumerate(valDataloader):
        st_time = time.time()
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask,scale,index_division = data
        # print(hist.shape)
        # print(nbrs.shape)
        # print(mask.nonzero().shape)

        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()
            scale = scale.cuda()

        # Forward pass
        if args['use_maneuvers']:
            if epoch_num < pretrainEpochs:
                # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
                net.train_flag = True
                fut_pred, _ , _ = net(hist, nbrs, lat_enc, lon_enc,index_division)
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                # During training with NLL loss, validate with NLL over multi-modal distribution
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, lat_enc, lon_enc,index_division)
                l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,avg_along_time = True)
                avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            fut_pred = net(hist, nbrs, lat_enc, lon_enc,index_division)
            fut = fut*scale
            fut_pred[:,:,0:2] = fut_pred[:,:,0:2]*scale
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

        avg_val_loss += l.item()
        val_batch_count += 1

    sio.savemat('val_data.mat',{'real':fut.detach().cpu().numpy(),'pred':fut_pred.detach().cpu().numpy()})

    # Print validation loss and update display variables
    print('Validation loss :',format(avg_val_loss/val_batch_count,'0.4f'),"| Val Acc:",format(avg_val_lat_acc/val_batch_count*100,'0.4f'),format(avg_val_lon_acc/val_batch_count*100,'0.4f'))
    val_loss.append(avg_val_loss/val_batch_count)
    prev_val_loss = avg_val_loss/val_batch_count
    print("current loss: ", avg_val_loss, ' history best loss: ', best_val_loss)
    if avg_val_loss < best_val_loss or epoch_num == 0:
        best_val_loss = avg_val_loss
        torch.save(net.state_dict(), 'trained_models/cslstm_ngsim_nei1_bestval.tar')
        print("Best val loss updated! current best val loss is at epoch: ", epoch_num + 1)
    #__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________



