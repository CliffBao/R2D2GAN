from __future__ import print_function
import torch
from model_wcgan_linear_socconv import highwayNet_d, highwayNet_g_single, highwayNet_g_compose
from utils_wcgan_linear_socconv import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest,mmd_mine

from torch.utils.data import DataLoader
import scipy.io as sio
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os
plt.switch_backend('agg')

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

## Network Arguments
args_d = {}
args_g={}
args_d['use_cuda'] = True
args_g['use_cuda'] = True

args_d['in_length'] = 81
args_g['in_length'] = 81
args_d['out_length'] = 81
args_g['out_length'] = 81

args_g['grid_size'] = (13,3)
args_d['grid_size'] = (13,3)
args_d['encoder_size'] = 128
args_d['input_embedding_size_d'] = 32
args_d['pool_embedding_size_d'] = 32
args_g['input_embedding_size_g'] = 64
args_g['encoder_size'] = 256
args_d['batch_size'] = 256
args_g['batch_size'] = 256

args_g['soc_conv_depth'] = 128
args_g['conv_3x1_depth'] = 32
args_g['train_flag'] = True
# Initialize network

## Initialize optimizer
# train generator every n times
trainEpochs = 100
lambda_gp = 10
n_critic = 1
learning_rate_d = 0.0004
learning_rate_g = 0.0001
b1 = 0
b2 = 0.90
# clip_value = 0.01
d_loss_record = []
g_loss_record = []
d_fake_loss_record = []
d_real_loss_record = []
d_penalty_loss_record = []
## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf


def diff(data_, order):
    for m in range(order):
        data_ = data_[:, :, 1:] - data_[:, :, :-1]
    data_ = torch.sqrt(torch.sum(data_ * data_, 1))
    # normalise into [-1,1]
    # data_min, _ = torch.min(data_, 1)
    # data_max, _ = torch.max(data_, 1)
    # data_min = data_min.view(data_.shape[0], 1)
    # data_max = data_max.view(data_.shape[0], 1)
    # data_ = (data_ - data_min) / (data_max - data_min)
    # implement first or first two value
    data_imple = data_[:,0].contiguous().view(data_.shape[0],1)
    data_imple = data_imple.repeat(1,order)
    data_ = torch.cat((data_imple, data_), 1)
    data_ = data_.unsqueeze(1)
    return data_
# torch.autograd.set_detect_anomaly(True)

def compute_gradient_penalty(D, real_samples, fake_samples,mask):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # sample shape:(time_length, batch_size*scene_traj_num, dim) = (81, 128, 2)
    real_samples = real_samples.permute(1,2,0)
    fake_samples = fake_samples.permute(1,2,0)
    # (batch_size*scene_traj_num,2,time_length)
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # scene_vel = diff(interpolates, 1)
    # scene_acc = diff(interpolates, 2)
    # interpolates = torch.cat((interpolates, scene_vel, scene_acc), 1)
    interpolates = interpolates.permute(2, 0, 1)

    d_interpolates = D(interpolates,mask)
    # out_shape = sum([len(x)*(len(x)+1)//2 for x in mask])
    out_shape = real_samples.shape[0]
    fake = Variable(Tensor(out_shape,1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    # with torch.autograd.detect_anomaly():
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    # b = torch.nn.utils.clip_grad_norm_(gradients, 10)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    # if torch.sum(torch.isnan(gp)):
    #     tmp = gradients.norm(2, dim=1)
    #     index = torch.where(torch.isnan(tmp)==1)
    #     print(tmp)
    #     print(index)
    #     print(interpolates[:,index[0].cpu().numpy(),:].permute(1,0,2))
    #     print('result',d_interpolates)
    #     print('gradients',gradients.shape,gradients.norm(2, dim=1),gradients)
    #     print('gp',gp)
    return gp

sample_interval = 1000
image_id = 0
# we first generate single trajectory dataset, then we load it to speed up compose training
# net_g = highwayNet_g_single(args_g)
# if args_g['use_cuda']:
#     net_g = net_g.cuda()
# net_g.load_state_dict(torch.load('trained_models/ngsim_wcgenerator_single_selu_full_d1000epoch_cond.tar'))
# for first run, set gen_set_name = None
# gen_set_name = 'data/Trainset_ngsim_clstm_smooth_wcgan_single_manuever6_fake.mat'
# gen_set_name = None
# args_g['neigh_num'] = 6
real_set_name = 'data/TrainSet-us1011-gan-smooth.mat'
hero_grid_seq = 19
hero_grid = (args_d['grid_size'][1]//2,args_d['grid_size'][0]//2)
trSet = ngsimDataset(mat_file=real_set_name, enc_size=args_g['encoder_size'],
                     hero_grid_pos = hero_grid_seq, t_f=args_d['in_length'],
                     batch_size=args_d['batch_size'], length = args_d['out_length'],
                     local_normalise=1)

trDataloader = DataLoader(trSet,batch_size=args_d['batch_size'],shuffle=True,num_workers=4,
                          drop_last = True, collate_fn=trSet.collate_fn)

net_d_compose = highwayNet_d(args_d)
net_g_compose = highwayNet_g_compose(args_g)

# we can start from last saved models
# net_d_compose.load_state_dict(torch.load('trained_models/ngsim_wdiscriminator_compose_100epoch.tar'))
# net_g_compose.load_state_dict(torch.load('trained_models/ngsim_wgenerator_compose_100epoch.tar'))

if args_d['use_cuda']:
    net_d_compose = net_d_compose.cuda()
if args_g['use_cuda']:
    net_g_compose = net_g_compose.cuda()

# clip_value = 0.01
d_loss_record = []
g_loss_record = []
d_fake_loss_record = []
d_real_loss_record = []
d_penalty_loss_record = []
# we implement WGAN-GP instead
# optimizer_d = torch.optim.RMSprop(net_d_compose.parameters(), lr = learning_rate_d)
# optimizer_g = torch.optim.RMSprop(net_g_compose.parameters(), lr = learning_rate_g)
optimizer_d = torch.optim.Adam(net_d_compose.parameters(), lr = learning_rate_d, betas = (b1,b2))
optimizer_g = torch.optim.Adam(net_g_compose.parameters(), lr = learning_rate_g, betas = (b1,b2))
Tensor = torch.cuda.FloatTensor if args_g['use_cuda'] else torch.FloatTensor
batch_size = args_d['batch_size']

## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf

mmd_record = np.zeros((trainEpochs))
mmd_epoch = np.zeros((trSet.__len__()//args_d['batch_size']))

# checkpoint part
start_epoch = 0
resume = True
if resume:
    if os.path.isfile('checkpoint.pkl'):
        checkpoint = torch.load('checkpoint.pkl')
        start_epoch = checkpoint['epoch'] + 1
        mmd_record = checkpoint['mmd_record']
        net_g_compose.load_state_dict(checkpoint['model_generator'])
        net_d_compose.load_state_dict(checkpoint['model_discriminator'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found")

change_count = 0
# checkpoint part end

t_clipped = args_g['in_length']
for start_epoch in range(start_epoch,trainEpochs):
    print('training epoch %d-%d'%(start_epoch,trainEpochs))
    epoch_start_t = time.time()
    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net_d_compose.train_flag = True
    net_g_compose.train_flag = True
    # Variables to track training performance:
    avg_tr_loss = 0
    avg_tr_time = 0

    for i, data in enumerate(trDataloader):

        st_time = time.time()
        real_scene,fake_scene,fake_scene_2,mask,index_division,maneuver,scale = data

        if args_d['use_cuda']:
            real_scene = real_scene[0:t_clipped,:,:].cuda()
            mask = mask.cuda()
            mask_traj = mask[:,:,:,0:2*t_clipped]
        if args_g['use_cuda']:
            fake_scene = fake_scene[0:t_clipped,:,:].cuda()
            fake_scene_2 = fake_scene_2[0:t_clipped,:,:].cuda()
            maneuver = maneuver.cuda()
            scale = scale.to(torch.float32).cuda()

        # generated shape:batch_size*grid[1]*grid[2]*(2*length)
        # print('before',fake_scene)
        composed_scene = net_g_compose(fake_scene,mask,maneuver)
        # get mask, reshape into batch_size*2*length
        composed_scene = composed_scene[mask_traj].view(-1,2,t_clipped)
        composed_scene = composed_scene.permute(2,0,1)

        # print('after',composed_scene)
        if args_d['use_cuda']:
            composed_scene = composed_scene.cuda()
            composed_scene = composed_scene*scale

        # mmd_tmp = mmd_mine(real_scene.permute(1,2,0).contiguous().view(-1,2*args_d['in_length']).detach(),
        #             composed_scene.permute(1,2,0).contiguous().view(-1,2*args_d['in_length']).detach())
        # mmd_epoch[i] = mmd_tmp
        # torch.cuda.empty_cache()
        # print('cache empty')
        # composed_scene = composed_scene.permute(1, 2, 0)
        # scene_vel = diff(composed_scene, 1)
        # scene_acc = diff(composed_scene, 2)
        # composed_scene = torch.cat((composed_scene, scene_vel, scene_acc), 1)
        # composed_scene = composed_scene.permute(2, 0, 1)
        #
        # real_scene = real_scene.permute(1, 2, 0)
        # scene_vel = diff(real_scene, 1)
        # scene_acc = diff(real_scene, 2)
        # real_scene = torch.cat((real_scene, scene_vel, scene_acc), 1)
        # real_scene = real_scene.permute(2, 0, 1)
        d_fake_loss = torch.mean(net_d_compose(composed_scene,index_division))
        d_real_loss = -torch.mean(net_d_compose(real_scene,index_division))

        gradient_penalty = compute_gradient_penalty(net_d_compose, real_scene, composed_scene, index_division)
        d_loss = d_fake_loss+d_real_loss
        d_loss = d_fake_loss+d_real_loss+lambda_gp*gradient_penalty
        # nan_flag = torch.sum(torch.isnan(d_loss))
        # if nan_flag:
        #     print(d_fake_loss,d_real_loss,gradient_penalty)
        #     print(composed_scene)
        #     print(real_scene)

        # assert nan_flag==0
        optimizer_d.zero_grad()
        # with torch.autograd.detect_anomaly():
        d_loss.backward()
        a = torch.nn.utils.clip_grad_norm_(net_d_compose.parameters(), 10)
        optimizer_d.step()
        # for p in net_d_compose.parameters():
        #     p.data.clamp_(-clip_value, clip_value)

        if i%n_critic==0:
            new_fake_sample = fake_scene_2
            raw_fake_sample = net_g_compose(new_fake_sample,mask,maneuver)
            fake_sample = raw_fake_sample[mask_traj].view(-1, 2, t_clipped)
            fake_sample = fake_sample.permute(2, 0, 1)
            if args_d['use_cuda']:
                fake_sample = fake_sample.cuda()
                fake_sample = fake_sample*scale
            # fake_sample = fake_sample.permute(1, 2, 0)
            # scene_vel = diff(fake_sample, 1)
            # scene_acc = diff(fake_sample, 2)
            # fake_sample = torch.cat((fake_sample, scene_vel, scene_acc), 1)
            # fake_sample = fake_sample.permute(2, 0, 1)
            # print(i,d_fake_loss,d_real_loss,gradient_penalty,real_scene)
            # assert torch.sum(torch.isnan(d_loss))==0

            g_loss = -torch.mean(net_d_compose(fake_sample,index_division))
            optimizer_g.zero_grad()
            # with torch.autograd.detect_anomaly():
            g_loss.backward()
            optimizer_g.step()
            d_loss_record.append(d_loss.detach().cpu().numpy())
            g_loss_record.append(g_loss.detach().cpu().numpy())
            d_fake_loss_record.append(d_fake_loss.detach().cpu().numpy())
            d_real_loss_record.append(d_real_loss.detach().cpu().numpy())
            # d_penalty_loss_record.append(gradient_penalty.detach().cpu().numpy())

        # Track average train loss and average train time:
        batch_time = time.time()-st_time
        avg_tr_loss += g_loss.item()
        avg_tr_time += batch_time
     
        if i%sample_interval==0:
            # raw fake sample: batch_size*3*13*161
            sample = raw_fake_sample.detach().cpu().numpy()
            plt.ion()
            draw_num = 1
            for draw in range(draw_num):
                tmp_mask = torch.zeros_like(mask_traj)
                tmp_mask[draw,:,:,:] = mask_traj[draw,:,:,:]
                draw_sample = sample[tmp_mask.cpu().numpy()].reshape(-1,2,t_clipped)

                for j in range(int(draw_sample.shape[0])):
                    plt.plot(draw_sample[j,0,:], draw_sample[j,1,:])

                hero_mask = torch.zeros_like(mask_traj)
                hero_mask[draw,hero_grid[0],hero_grid[1],:] = mask_traj[draw,hero_grid[0],hero_grid[1],:]
                hero_traj = sample[hero_mask.cpu().numpy()].reshape(2,t_clipped)
                plt.text(hero_traj[0,0],hero_traj[1,0],'hero')

                path = 'gen_samples/full_compose/'
                fullname = np.char.add(path,str(image_id+draw))
                plt.savefig(fullname.item())
                plt.show()
                plt.close()
            # fake_sample_checkpoint = np.zeros(draw_sample.shape)
            # fake_sample_checkpoint[:] = draw_sample[:]
            # real scene: 81*(batch*scene_traj_num)*2
            sample = real_scene.detach().permute(1,2,0)
            sample = sample.contiguous().view(sample.shape[0],sample.shape[1]*sample.shape[2])
            soc_enc = torch.zeros_like(mask_traj).float()
            soc_enc = soc_enc.masked_scatter_(mask_traj, sample)
            sample = soc_enc.cpu().numpy()
            for draw in range(draw_num):
                tmp_mask = torch.zeros_like(mask_traj)
                tmp_mask[draw, :, :, :] = mask_traj[draw, :, :, :]
                draw_sample = sample[tmp_mask.cpu().numpy()].reshape(-1, 2, t_clipped)
                # print(draw_sample.shape)
                # print(draw_sample)
                for j in range(int(draw_sample.shape[0])):
                    plt.plot(draw_sample[j, 0, :], draw_sample[j, 1, :])
                hero_mask = torch.zeros_like(mask_traj)
                hero_mask[draw, hero_grid[0], hero_grid[1], :] = mask_traj[draw, hero_grid[0], hero_grid[1], :]
                hero_traj = sample[hero_mask.cpu().numpy()].reshape(2, t_clipped)
                plt.text(hero_traj[0, 0], hero_traj[1, 0], 'hero')
                plt.scatter(hero_traj[0, 0], hero_traj[1, 0])
                path = 'real_samples/full_compose/'
                fullname = np.char.add(path, str(image_id + draw))
                plt.savefig(fullname.item())
                plt.show()
                plt.close()
            # real_sample_checkpoint = np.zeros(draw_sample.shape)
            # real_sample_checkpoint[:] = draw_sample[:]
            # sample_checkpoint_name = np.char.add('sample_data/',str(image_id+draw_num))
            # sio.savemat(np.char.add(sample_checkpoint_name,'.mat').item(),{'fake':fake_sample_checkpoint,'real':real_sample_checkpoint})
            # print('%d mat saved!'%(image_id+draw_num))
            image_id = image_id+sample_interval

        if i%100 == 99:
            print("Compose Traj Training Epoch no:",start_epoch+1," d_loss:",d_loss.detach().cpu().numpy(),
                  ' d_fake_loss:',d_fake_loss.detach().cpu().numpy(),' d_real_loss:',d_real_loss.detach().cpu().numpy(),
                  ' g_loss', g_loss.detach().cpu().numpy(), ' avg g loss: ', avg_tr_loss, ' avg train time:',avg_tr_time)
            train_loss.append(avg_tr_loss/100)
            avg_tr_loss = 0
            avg_tr_time = 0
    epoch_end_t = time.time()
    mmd_record[start_epoch] = np.mean(mmd_epoch)
    print("epoch no:",start_epoch+1," spent total time is:", epoch_end_t-epoch_start_t,' ave mmd:', mmd_record[start_epoch])
     # checkpoint,save every 5 epochs
    if start_epoch%3==0:
        checkpoint = {
            'epoch': start_epoch,
            'model_generator': net_g_compose.state_dict(),
            'model_discriminator': net_d_compose.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'mmd_record': mmd_record
        }
        # torch.cuda.empty_cache()
        torch.save(checkpoint, 'checkpoint.pkl')
        print('checkpoint saved')
    #
    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
torch.save(net_g_compose.state_dict(), 'trained_models/ngsim_wcgenerator_compose_100epoch_us1011_rpn.tar')
torch.save(net_d_compose.state_dict(), 'trained_models/ngsim_wcdiscriminator_compose_100epoch_us1011_rpn.tar')
sio.savemat('wcgan_compose_loss_100_us1011_rpn.mat',{'d_loss':d_loss_record,'g_loss':g_loss_record,
                             'd_fake_loss':d_fake_loss_record,'d_real_loss':d_real_loss_record})
sio.savemat('mmd_record.mat',{'mmd':mmd_record})
