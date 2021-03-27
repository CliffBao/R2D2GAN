# Multiple agents' spatiotemporal data generation based on recurrent regression dual discriminator GAN

## Abstract 

Generative Adversarial Networks (GANs) have proven its capability of generating realistic-looking data and have been widely used in image related and time-series applications. However, generation of multiple agents’ spatiotemporal data remains an unexplored region. In this work, we propose a recurrent regression dual discriminator GAN named R2D2GAN. A novel generator is designed to learn mappings from prior stochastic process samplings to multiple agents’ spatiotemporal data, which conditions on spatial configuration of multiple agents only. A classification discriminator and a regression discriminator are proposed to represent different features of spatiotemporal data. The classification discriminator learn to represent spatial and sequential features of each agent. The regression discriminator learn to represent inherent sequential dependency for target agent. To stabilize training of GAN, a min-max game is elaborately designed and new training losses are proposed for dual discriminators and the generator. To validate learning ability of proposed R2D2GAN, we embed it in vehicle trajectory prediction application. Through qualitative and quantitative evaluation methods, we show that the R2D2GAN is capable of generating realistic-looking multiple agents’ spatiotemporal data with acceptable performance degradation in prediction task.

## Introduction

This is source code repository for *Multiple agents' spatiotemporal data generation based on recurrent regression dual discriminator GAN*. Some large data files over 100MB is uploaded to my google drive at https://drive.google.com/drive/folders/1dBOOvOPZ2nqQwGbleBN6Cbk7g9XuKvui?usp=sharing, as some problems encountered when using github lfs.

## Runtime environment 

Python 3.6

## File description

### R2D2GAN

This is our R2D2GAN training program. Running *python3 train_wcgan_decompose.py*  should train a generative model for NGSIM dataset. We save generation model every 3 epoch. Training dataset configuration is in line 197. 

*Note: There are some code snippets that are not used actually!*

### R2D2GAN/data

Folder contains training dataset.

### R2D2GAN/927.3try/rdgenerator_fm

This folder contains related files of a experiment where generator loss includes feature matching (FM) loss and adversarial loss. 

*./gen_samples*: Contains generated prediction scenarios (multiple agents' trajectories in one scene) while training is going on. 

*./trained_models*: Contains trained models while training. 

*net_io.mat*: Contains inputs and outputs of a generator in one epoch, and corresponding real ones. 

​                   *index_div* is a cell of index array that indicates trajectory ids that in one prediction scenario. 

​					*pos_condition* is another input of the generator. 

​					*composed_scene* is outputs of the generator where one row represents one trajectory. If viewed in matlab, each row should reshape into (81,2) as 												  matlab index is different from python.

​					*real_scene* is real trajectories corresponding to *pos_condition*..

*checkpoint.ckl*: Checkpoint file.

*loss.mat:* Loss records of each components in R2D2GAN.

### R2D2GAN/927.3try/generator_nofm

This folder contains related files of a experiment where generator loss includes adversarial loss only. 

Others are the same as above.

### R2D2GAN/927.3.1try

This folder contains related files of a experiment where regression discriminator is no longer used and generator loss still includes feature matching (FM) loss and adversarial loss. 

Others are the same as above

### R2D2GAN/927.3.2try

This folder contains related files of a experiment where generator loss includes original FM loss only. 

Others are the same as above.