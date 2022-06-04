# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 23:35:00 2022

@author: yangyue
"""

import os
import glob
import sys
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np


from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam

# project imports
sys.path.append('../ext/medipy-lib')

import network_DE
import losses
from tensorboardX import SummaryWriter
import nibabel as nib
import random

writer = SummaryWriter('../logs/')
vol_size = (160,192,160)                
base_data_dir = '../../data/OASIS1_MY/'  
atlas_dir = '../atlas/'


train_vol_names =  glob.glob(base_data_dir + 'train/*')
train_seg = glob.glob(base_data_dir +'train_seg/*')

train_vol_names = sorted(train_vol_names)
train_seg = sorted(train_seg)

atlas = nib.load(base_data_dir + 'OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.nii.gz')
atlas_vol = atlas.get_fdata()
affine = atlas.affine.copy()
atlas_vol = np.reshape(atlas_vol, (1,) + atlas_vol.shape+(1,))


atlas_seg = nib.load(base_data_dir + 'OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.nii.gz') 
atlas_vol_seg = atlas_seg.get_data()
atlas_vol_seg = np.reshape(atlas_vol_seg, (1,) + atlas_vol_seg.shape+(1,))


atlas_CSF = nib.load(atlas_dir + 'CSF_OASIS1_de_3.nii')
atlas_CSF = atlas_CSF.get_fdata()

atlas_GM = nib.load(atlas_dir + 'GM_OASIS1_de_1.nii')
atlas_GM = atlas_GM.get_fdata()

atlas_WM = nib.load(atlas_dir + 'WM_OASIS1_de_5.nii')
atlas_WM = atlas_WM.get_fdata()

atlas_CSF = np.reshape(atlas_CSF, (1,) + atlas_CSF.shape+(1,))
atlas_GM = np.reshape(atlas_GM, (1,) + atlas_GM.shape+(1,))
atlas_WM = np.reshape(atlas_WM, (1,) + atlas_WM.shape+(1,))



weights = [1.0, 1.0, 0.3, 0.3, 0.3]

def train(model,save_name, gpu_id, lr, n_iterations, reg_param, model_save_iter):

    model_dir = '../model_DE_OASIS1_0.9/' #模型保存目录
    
    #如果目录不存在，建立文件夹
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    gpu = '/gpu:' + str(gpu_id)   #配置gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # UNET filters
    nf_enc = [16,32,32,32]    #编码器
    if(model == 'vm1'):
        nf_dec = [32,32,32,32,8,8,3]   #解码器
    else:
        nf_dec = [32,32,32,32,32,16,16]

    with tf.device(gpu):
        model = network_DE.voxel_SDF_net(vol_size, nf_enc, nf_dec)

        model.summary()
 
        model.compile(optimizer=Adam(lr=lr),
                      loss=[losses.cc3D(), losses.gradientLoss('l2'),
                            losses.MSE(),losses.MSE(),losses.MSE()], 
                      loss_weights=weights)


    zero_flow = np.zeros((1, vol_size[0], vol_size[1], vol_size[2], 3))
    
    l = list(range(len(train_vol_names)))
    
    for step in range(0, n_iterations):
        
        random.shuffle(l) 
        
        for i in range(len(train_vol_names)):
            
            a = l[i]
        
            X = nib.load(train_vol_names[a])                #读取那个图片              
            X = X.get_fdata()                               #取出值
            X = np.reshape(X, (1,) + X.shape + (1,))        #改变维度
            
            X_seg = nib.load(train_seg[a])                  #读取分割图片              
            X_seg = X_seg.get_fdata()                       #取出值
            
            X_CSF = X_seg == 1                              #取出训练图像脑脊液的分割图
            X_CSF = X_CSF.astype(np.float32)                #将 bool 转换成 float32
            
            X_GM = X_seg == 2
            X_GM = X_GM.astype(np.float32)
            
            X_WM = X_seg == 3
            X_WM = X_WM.astype(np.float32)
            
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))    #改变维度
            X_CSF = np.reshape(X_CSF, (1,) + X_CSF.shape + (1,)) 
            X_GM = np.reshape(X_GM, (1,) + X_GM.shape + (1,)) 
            X_WM = np.reshape(X_WM, (1,) + X_WM.shape + (1,)) 
            
            train_loss = model.train_on_batch([X, atlas_vol,X_CSF,X_GM,X_WM], 
                [atlas_vol,zero_flow,
                 atlas_CSF,atlas_GM,atlas_WM,

                 ])

          
            if not isinstance(train_loss, list):
                train_loss = [train_loss]
    
            printLoss(step, i, train_loss)
            
        
        if(step % model_save_iter == 0):
            model.save(model_dir + '/' + str(step) + '.h5')
            writer.add_scalar('0', train_loss[0],step)
            writer.add_scalar('1', train_loss[1],step)
            writer.add_scalar('2', train_loss[2],step)

 


def printLoss(step, training, train_loss):
    s = str(step) + "," + str(training)

    if(isinstance(train_loss, list) or isinstance(train_loss, np.ndarray)):
        for i in range(len(train_loss)):
            s += "," + str(train_loss[i])
    else:
        s += "," + str(train_loss)

    print(s)
    

    
   
    sys.stdout.flush()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str,dest="model", 
                        choices=['vm1','vm2'],default='vm2',
                        help="Voxelmorph-1 or 2")
    '''parser.add_argument("--save_name", type=str,required=False,
                        dest="save_name", help="Name of model when saving")'''
    parser.add_argument("--save_name", type=str,# required=True,
                        dest="save_name", default='defaultsavedmodel',
                        help="Name of model when saving")
    parser.add_argument("--gpu", type=int,default=1,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float, 
                        dest="lr", default=1e-4,help="learning rate") 
    parser.add_argument("--iters", type=int, 
                        dest="n_iterations", default=1000,
                        help="number of iterations")
    parser.add_argument("--lambda", type=float, 
                        dest="reg_param", default=1.0,
                        help="regularization parameter")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=10, 
                        help="frequency of model saves")

    args = parser.parse_args()
    train(**vars(args))
    
#    reduction_ops_common.h:155 : Invalid argument: Invalid reduction dimension (5 for input with 5 dimension(s)

