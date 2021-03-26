from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
import time
import scipy.io as sio
import numpy

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32

# Evaluation metric:
# metric = 'nll'  #or rmse
metric = 'rmse'

# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('trained_models/cslstm_us1011_nei3&4_norminloss_bestval.tar'))
if args['use_cuda']:
    net = net.cuda()

tsSet = ngsimDataset('data/TestSet-us1011-gan-smooth-nei3&4.mat')
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=4,collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(25).cuda()
counts = torch.zeros(25).cuda()


for i, data in enumerate(tsDataloader):
    st_time = time.time()
    hist, nbrs, fut, op_mask,scale,index_div = data

    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()
        scale = scale.cuda()

    fut_pred = net(hist, nbrs, index_div)
    fut = fut*scale
    fut_pred[:, :, 0:2] = fut_pred[:, :, 0:2] * scale
    l, c = maskedMSETest(fut_pred, fut, op_mask)
    if i == 0:
        pred = torch.reshape(fut_pred, (25, fut_pred.shape[1]*fut_pred.shape[2]))
        real = torch.reshape(fut, (25, fut.shape[1]*fut.shape[2]))
        pred = pred.cpu().detach().numpy()
        real = real.cpu().detach().numpy()

        sio.savemat('cslstm-eval.mat',
                    {'predict': pred, 'fut': real})
    # print(l)
    lossVals +=l.detach()
    counts += c.detach()

if metric == 'nll':
    print(lossVals / counts)
else:
    print(torch.pow(lossVals / counts,0.5)*0.3048)   # Calculate RMSE and convert from feet to meters


