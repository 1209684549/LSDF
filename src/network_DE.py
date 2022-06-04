
#DESDF + NJLOSS

import sys


# third party
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D,Add,Dropout,BatchNormalization,MaxPooling3D,concatenate,add,Multiply, multiply
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.layers import Subtract
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf

# import neuron layers, which will be useful for Transforming.
sys.path.append('../ext')
sys.path.append('../ext/neuron')
sys.path.append('../ext/neuron/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools')
#sys.path.append('../ext/pytools-lib')
import neuron.layers as nrn_layers

def getBinaryTensor(imgTensor, boundary = 1):
    
    one = tf.ones_like(imgTensor)
    zero = tf.zeros_like(imgTensor)
    return tf.where(imgTensor > boundary, one, zero)


def conv_dilation(image,n):   #image:需要膨胀的数据   n:膨胀区域的标签
    
    W = tf.constant(value=np.ones((3,3,3), dtype=np.float32), shape=(3,3,3,1,1))
    image1 = tf.nn.conv3d(image, W, strides=[1, 1, 1, 1,1], padding='SAME')
    image2 = getBinaryTensor(image1,1)   #二值化后的膨胀图像
    d1 = tf.subtract(image2,image)       #膨胀的部分
    d1 = tf.multiply(d1,n)
    
    return d1,image2

def conv_erosion(image,n):   #image:需要膨胀的数据   n:膨胀区域的标签

    W = tf.constant(value=np.ones((3,3,3), dtype=np.float32), shape=(3,3,3,1,1))
    image1 = tf.nn.conv3d(image, W, strides=[1, 1, 1, 1,1], padding='SAME')
    image2 = getBinaryTensor( image1,26.9)
    d1 = tf.subtract(image,image2)       #膨胀的部分
    d1 = tf.multiply(d1,n)
    
    return d1,image2

    
def SDF_GM(data):   # GM 1次
    
    d1,data1 = conv_dilation(data,-0.5)
    d111,data11 = conv_erosion(data,0.5)

    
    p = data1 - 1
    p = tf.multiply(p,1.5)
    
    e = tf.multiply(data11,1.5)

    result = d1 + d111  + p + e
    return result
    

def SDF_CSF(data):   # CSF 3次
    
    d1,data1 = conv_dilation(data,-0.5)
    d2,data2 = conv_dilation(data1,-1.5)
    d3,data3 = conv_dilation(data2,-2.5)
    
    d111,data11 = conv_erosion(data,0.5)
    d21,data21 = conv_erosion(data11,1.5)
    d31,data31 = conv_erosion(data21,2.5)
 

    p = data3 - 1
    p = tf.multiply(p,3.5)
    
    e = tf.multiply(data31,3.5)
    
    result =  d1 + d111 + d2 + d21 + d3 + d31 +p + e

    return result


def SDF_WM(data):  # WM 5次
    
    d1,data1 = conv_dilation(data,-0.5)
    d2,data2 = conv_dilation(data1,-1.5)
    d3,data3 = conv_dilation(data2,-2.5)
    d4,data4 = conv_dilation(data3,-3.5)
    d5,data5 = conv_dilation(data4,-4.5)

    

    d111,data11 = conv_erosion(data,0.5)
    d21,data21 = conv_erosion(data11,1.5)
    d31,data31 = conv_erosion(data21,2.5)
    d41,data41 = conv_erosion(data31,3.5)
    d51,data51 = conv_erosion(data41,4.5)
    
    p = data5 - 1
    p = tf.multiply(p,5.5)
    e = tf.multiply(data51,5.5)

    
    result = d1 + d111 + d2 + d21 + d3 + d31 + d4 + d41 + d5 + d51  + p + e
    
    return result



def voxel_SDF_net(vol_size, enc_nf, dec_nf,full_size=True, indexing='ij'):
    
    '''
    用固定卷积操作实现SDF
    '''

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    unet_model1 = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size)
    [src,tgt,CSF,GM,WM] = unet_model1.inputs
    x1,CSF,GM,WM = unet_model1.output

    # transform the results into a flow field.
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow1 = Conv(ndims, kernel_size=3, padding='same', name='flow1',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x1)
    
    
    # warp the source with the flow
    y1 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow1])
    CSF1 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([CSF, flow1])
    GM1 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([GM, flow1])
    WM1 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([WM, flow1])
    
    csf = Lambda(SDF_CSF,output_shape=(160,192,160,1))(CSF1)
    gm = Lambda(SDF_GM,output_shape=(160,192,160,1))(GM1)
    wm = Lambda(SDF_WM,output_shape=(160,192,160,1))(WM1)
    
    

    # prepare model
    model = Model(inputs=[src,tgt,CSF,GM,WM], outputs=[y1,flow1,
                                                       csf,gm,wm,
                                                       ])
    
    return model



def unet_core(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None,
              CSF = None,WM = None,GM = None,src_feats=1, tgt_feats=1):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    # inputs
    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
        
    if CSF is None:
        CSF = Input(shape=[*vol_size, tgt_feats])
    if GM is None:
        GM = Input(shape=[*vol_size, tgt_feats])
    if WM is None:
        WM = Input(shape=[*vol_size, tgt_feats])

        
    x_in = concatenate([src, tgt])
    

    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])
    
    # only upsampleto full dim if full_size
    # here we explore architectures where we essentially work with flow fields 
    # that are 1/2 size 
    if full_size:
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[0]])
        x = conv_block(x, dec_nf[5])

    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])

    return Model(inputs=[src,tgt,CSF,GM,WM], outputs=[x,CSF,GM,WM])



def conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

