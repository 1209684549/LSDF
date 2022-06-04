
# Third party inports
import tensorflow as tf
import numpy as np
import sys
# Third party inports

import keras.backend as K

sys.path.append('../ext')
sys.path.append('../ext/neuron')
import labelreg.losses as labelloss
import neuron.layers as nrn_layers
#import ext.labelreg.losses as labelloss

def Get_Ja(displacement):

    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''

    D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])



    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])

    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])

    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

    return D1-D2+D3

def NJ_loss(): 
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    def loss(y_true,ypred):

        Neg_Jac = 0.5*(tf.abs(Get_Ja(ypred)) - Get_Ja(ypred))
        return tf.reduce_sum(Neg_Jac)
    return loss


def dice_coe( loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    '''
    计算 dice Loss
    '''
    def loss(output, target):

        inse = tf.reduce_sum(output * target, axis=axis)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output, axis=axis)
            r = tf.reduce_sum(target * target, axis=axis)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output, axis=axis)
            r = tf.reduce_sum(target, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)
        return -dice
    return loss

def dice_coe_single(output,target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    

    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return -dice


def multi_scale_loss():
    def loss(label_fixed, label_moving,loss_scales = [0, 1, 2, 4, 8, 16]):
        label_loss_all = tf.stack(
            [dice_coe_single(
                separable_filter3d(label_fixed, gauss_kernel1d(s)),
                separable_filter3d(label_moving, gauss_kernel1d(s)))
                for s in loss_scales])
        return tf.reduce_mean(label_loss_all)
    return loss


def single_scale_loss():
    def loss(label_fixed, label_moving):
        label_loss_batch = 1 - dice_simple(label_fixed, label_moving)
        return label_loss_batch
    return loss

def dice_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts*ps, axis=[1, 2, 3]) * 2
    denominator = tf.reduce_sum(ts, axis=[1, 2, 3]) + tf.reduce_sum(ps, axis=[1, 2, 3, 4])+eps_vol
    return numerator/denominator


def diceLoss():

    def loss(y_true, y_pred):
        top = 2*tf.reduce_sum(y_true * y_pred, [1, 2, 3])
        bottom = tf.maximum(tf.reduce_sum(y_true+y_pred, [1, 2, 3]), 1e-5)
        dice = tf.reduce_mean(top/bottom)
        return -dice
    return loss

def MSE():
    def loss(y_true,y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return mse
    return loss

def gradientLoss(penalty='l1'):
    def loss(y_true, y_pred):
        dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
        return d/3.0

    return loss


def gradientLoss2D():
    def loss(y_true, y_pred):
        dy = tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

        dy = dy * dy
        dx = dx * dx

        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)
        return d/2.0

    return loss


def cc3D(win=[9, 9, 9], voxel_weights=None):
    def loss(I, J):
        I2 = I*I
        J2 = J*J
        IJ = I*J

        filt = tf.ones([win[0], win[1], win[2], 1, 1])

        I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]*win[2]
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var+1e-5)

        # if(voxel_weights is not None):
        #	cc = cc * voxel_weights

        return -1.0*tf.reduce_mean(cc)

    return loss


def cc2D(win=[9, 9]):
    def loss(I, J):
        I2 = tf.multiply(I, I)
        J2 = tf.multiply(J, J)
        IJ = tf.multiply(I, J)

        sum_filter = tf.ones([win[0], win[1], 1, 1])

        I_sum = tf.nn.conv2d(I, sum_filter, [1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv2d(J, sum_filter, [1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv2d(I2, sum_filter, [1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv2d(J2, sum_filter, [1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv2d(IJ, sum_filter, [1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]

        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
        return -1.0*tf.reduce_mean(cc)
    return loss
def GNetloss():
     def loss(y_true, y_pred):
          return  tf.log(1-y_pred)
     return loss
def binary_dice(y_true, y_pred):
    """
    N-D dice for binary segmentation
    """
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = 1 + tf.range(ndims)
#    vol_axes = 1 + np.arange(ndims)

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.maximum(tf.reduce_sum(y_true + y_pred, vol_axes), 1e-5)
#    dice = tf.reduce_mean(top/bottom)
#    return -dice
    return -tf.reduce_mean(top/bottom)


class NCC():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None,downsamplesize=None,eps=1e-5):
        self.win = win
        self.eps = eps
        self.downsamplesize = downsamplesize

    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
#        ndims = I.get_shape().as_list()
#        assert len(ndims) in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %s" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        if self.downsamplesize is not None:
            size = [self.downsamplesize]*5
            size[0]=size[4]=1
            I = tf.nn.max_pool3d(I,ksize = size,strides= size,padding='VALID')
        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)
    
    def loss(self, I, J):
        return - self.ncc(I, J)
class NCC_SSIM():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None,alpha=0.5, eps=1e-5):
        self.win = win
        self.alpha = alpha
        self.eps = eps


    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)
    
    def ssim3d(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        # Compute SSIM over tf.float32 Tensors.
#        im1 = tf.image.convert_image_dtype(I, tf.float32)
#        im2 = tf.image.convert_image_dtype(J, tf.float32)
#        ssim2 = tf.image.ssim(im1, im2, max_val=1.0)
        # ssim1 and ssim2 both have type tf.float32 and are almost equal.
        # set window size
        

        # return negative cc.
        return tf.reduce_sum(tf.image.ssim(I, J, 1.0))


    def loss(self, I, J):
#        return - self.ncc(I, J)
        return - self.alpha*self.ncc(I, J) - (1-self.alpha)*self.ssim3d(I, J)


def gauss_kernel1d(window_size=11,sigma=1.5):
    if sigma == 0:
        return 0
    else:
        k = tf.exp([-0.5*x**2/sigma**2 for x in range(-window_size//2, window_size//2+1)])
        return k / tf.reduce_sum(k)

def separable_filter3d(vol, kernel):
    if kernel == 0:
        return vol
    else:
        strides = [1, 1, 1, 1, 1]
        if vol.get_shape()[-1]==3:
            return tf.concat([tf.nn.conv3d(tf.nn.conv3d(tf.nn.conv3d(
                tf.expand_dims(vol[...,i],-1),
                tf.reshape(kernel, [-1, 1, 1, 1, 1]), strides, "SAME"),
                tf.reshape(kernel, [1, -1, 1, 1, 1]), strides, "SAME"),
                tf.reshape(kernel, [1, 1, -1, 1, 1]), strides, "SAME") for i in range(0,3)],-1)
        else:
            return tf.nn.conv3d(tf.nn.conv3d(tf.nn.conv3d(
                vol,
                tf.reshape(kernel, [-1, 1, 1, 1, 1]), strides, "SAME"),
                tf.reshape(kernel, [1, -1, 1, 1, 1]), strides, "SAME"),
                tf.reshape(kernel, [1, 1, -1, 1, 1]), strides, "SAME")

class SSIM3D():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, L=1.0, window_size=11):
        self.window_size = window_size
        self.k1 = 0.01
        self.k2 = 0.03
        self.data_range = L # maxvalue


    def ssim3d(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
#        ndims = len(I.get_shape().as_list()) - 2
#        assert ndims in [2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
#        sum_filt = tf.ones([*self.win, 1, 1])
#        strides = 1
#        if ndims > 1:
#            strides = [1] * (ndims + 2)
#        padding = 'SAME'
        mu1 = separable_filter3d(I, gauss_kernel1d())
        mu2 = separable_filter3d(J, gauss_kernel1d())
    
        mu1_sq  = tf.square(mu1)
        mu2_sq  = tf.square(mu2)
        mu1_mu2 = mu1 * mu2
    
        sigma1_sq = separable_filter3d(I*I, gauss_kernel1d()) - mu1_sq
        sigma2_sq = separable_filter3d(J*J, gauss_kernel1d()) - mu2_sq
        sigma12   = separable_filter3d(I*J, gauss_kernel1d()) - mu1_mu2
    
        C1 = (self.k1*self.data_range)**2
        C2 = (self.k2*self.data_range)**2
    
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        return tf.reduce_mean(ssim_map)

    def loss(self, I, J):
        return - self.ssim3d(I, J)
    
class NCC_SSIM3D():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None, L=1, window_size=11,alpha=0.5, eps=1e-5):
        self.win = win
        self.alpha = alpha
        self.eps = eps
        self.window_size = window_size
        self.k1 = 0.01
        self.k2 = 0.03
        self.data_range = L # maxvalue


    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)
    
    def ssim3d(self, I, J):
        mu1 = separable_filter3d(I, gauss_kernel1d())
        mu2 = separable_filter3d(J, gauss_kernel1d())
    
        mu1_sq  = tf.square(mu1)
        mu2_sq  = tf.square(mu2)
        mu1_mu2 = mu1 * mu2
    
        sigma1_sq = separable_filter3d(I*I, gauss_kernel1d()) - mu1_sq
        sigma2_sq = separable_filter3d(J*J, gauss_kernel1d()) - mu2_sq
        sigma12   = separable_filter3d(I*J, gauss_kernel1d()) - mu1_mu2
    
        C1 = (self.k1*self.data_range)**2
        C2 = (self.k2*self.data_range)**2
    
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        return tf.reduce_mean(ssim_map)

    def loss(self, I, J):
        return - self.alpha*self.ncc(I, J) - (1-self.alpha)*self.ssim3d(I, J)

class SSIM3D_simple():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=[11,11,11], eps=1e-5):
        self.win = win
        self.eps = eps


    def ssim3d(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
#        ndims = len(I.get_shape().as_list()) - 2
#        assert ndims in [2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        return tf.reduce_sum(tf.image.ssim(I, J, 1.0))

    def loss(self, I, J):
        return - self.ssim3d(I, J)
    
def ssim_simple(ts, ps):
    ssim_v = tf.reduce_sum(tf.image.ssim(ts, ps, 1.0))
    return ssim_v

def ssim_multiscal(ts, ps):
    ssim_v = tf.reduce_sum(tf.image.ssim_multiscale(ts, ps, 1.0))
    return ssim_v


class Grad():
    """
    N-D gradient loss
    """

    def __init__(self, penalty='l1'):
        self.penalty = penalty

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]
            
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)
        
        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            df = [tf.reduce_mean(f * f) for f in self._diffs(y_pred)]
        return tf.add_n(df) / len(df)


class Miccai2018():
    """
    N-D main loss for VoxelMorph MICCAI Paper
    prior matching (KL) term + image matching term
    """

    def __init__(self, image_sigma, prior_lambda, flow_vol_shape=None):
        self.image_sigma = image_sigma
        self.prior_lambda = prior_lambda
        self.D = None
        self.flow_vol_shape = flow_vol_shape


    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently, 
        has a '1' in the immediate neighbor, and 0 elsewehre.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """

        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied 
        # ith feature to ith feature
        filt = np.zeros([3] * ndims + [ndims, ndims])
        for i in range(ndims):
            filt[..., i, i] = filt_inner
                    
        return filt


    def _degree_matrix(self, vol_shape):
        # get shape stats
        ndims = len(vol_shape)
        sz = [*vol_shape, ndims]

        # prepare conv kernel
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # prepare tf filter
        z = K.ones([1] + sz)
        filt_tf = tf.convert_to_tensor(self._adj_filt(ndims), dtype=tf.float32)
        strides = [1] * (ndims + 2)
        return conv_fn(z, filt_tf, strides, "SAME")


    def prec_loss(self, y_pred):
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i

        Note: could probably do with a difference filter, 
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        vol_shape = y_pred.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)
        
        sm = 0
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y_pred, r)
            df = y[1:, ...] - y[:-1, ...]
            sm += K.mean(df * df)

        return 0.5 * sm / ndims


    def kl_loss(self, y_true, y_pred):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """

        # prepare inputs
        ndims = len(y_pred.get_shape()) - 2
        mean = y_pred[..., 0:ndims]
        log_sigma = y_pred[..., ndims:]
        if self.flow_vol_shape is None:
            # Note: this might not work in multi_gpu mode if vol_shape is not apriori passed in
            self.flow_vol_shape = y_true.get_shape().as_list()[1:-1]

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims, 
        # which is a function of the data
        if self.D is None:
            self.D = self._degree_matrix(self.flow_vol_shape)

        # sigma terms
        sigma_term = self.prior_lambda * self.D * tf.exp(log_sigma) - log_sigma
        sigma_term = K.mean(sigma_term)

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda * self.prec_loss(mean)

        # combine terms
        return 0.5 * ndims * (sigma_term + prec_term) # ndims because we averaged over dimensions as well


    def recon_loss(self, y_true, y_pred):
        """ reconstruction loss """
        return 1. / (self.image_sigma**2) * K.mean(K.square(y_true - y_pred))


class Lable_Loss():
    """
    similarity_type:'cross-entropy', 'mean-squared', 'dice', 'doubledice', 'doubledice-ssim', 'jaccard'
    similarity_scales:: [0, 1, 2, 4, 8, 16]
    regulariser_type: 'bending','gradient-l2','gradient-l1'
    """
    def __init__(self, vol_size = (144, 176, 112), similarity_type='doubledice', similarity_scales =[0, 1, 2, 4, 8, 16] ,regulariser_type='bending', regulariser_weight= 0.5):
        self.loss_type = similarity_type
        self.loss_scales = similarity_scales
        self.energy_type = regulariser_type
        self.energy_weight = regulariser_weight
        self.vol_size = vol_size

    def similarity_loss(self, Fixlabel, Movelabel):
        """ reconstruction loss """
        ndims = len(Movelabel.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        label_similarity = labelloss.multi_scale_loss(Fixlabel, Movelabel, self.loss_type, self.loss_scales)
        return tf.reduce_mean(label_similarity)

    def regulariser_loss(self, _, y_pred):
        ndims = len(y_pred.get_shape().as_list()) - 2
        assert ndims in [3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        """ reconstruction loss """
        loss_regulariser = labelloss.local_displacement_energy(y_pred, self.energy_type, self.energy_weight)
        return loss_regulariser
    def seg_loss(self, y_true, y_pred):
        """ reconstruction loss """
#        x = y_true.get_shape().as_list()
#        print("volumes should be 1 to 3 dimensions. found: %s" % x)
        y_trues = tf.split(y_true,num_or_size_splits=2, axis=-1)
        y_true0 = tf.reshape(y_trues[0], (1,*self.vol_size, 3))
        y_true1 = tf.reshape(y_trues[1], (1,*self.vol_size, 3))
        y_pred0 = tf.reshape(y_pred, (1,*self.vol_size, 3))
#        WMovelabel = nrn_utils.transform(y_true1, y_pred0, interp_method='linear', indexing='ij')
        WMovelabel = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([y_true1, y_pred0])
        y_pred1 = tf.reshape(WMovelabel, (1,*self.vol_size, 3))
        return self.similarity_loss(y_true0, y_pred1)+ self.energy_weight*self.regulariser_loss(y_pred, y_pred)
class RegiNet_Loss():
    """
    similarity_type:'cross-entropy', 'mean-squared', 'dice', 'doubledice', 'doubledice-ssim', 'jaccard'
    similarity_scales: [0, 1, 2, 4, 8, 16]
    regulariser_type: 'bending','gradient-l2','gradient-l1'
    """
    def __init__(self, similarity_type='doubledice', similarity_scales =[0, 1, 2, 4, 8, 16] ,regulariser_type='bending', downsamplesize = None):
        self.loss_type = similarity_type
        self.loss_scales = similarity_scales
        self.energy_type = regulariser_type
        self.downsamplesize = downsamplesize
        self.energy_weight = 1.0

    def similarity_loss(self, y_true, y_pred):
        """ reconstruction loss """
        ndims = len(y_pred.get_shape().as_list()) - 2
        if self.downsamplesize is not None:
            size = [self.downsamplesize]*5
            size[0]=size[4]=1
            y_true = tf.nn.max_pool3d(y_true,ksize = size,strides= size,padding='VALID')
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        label_similarity = labelloss.multi_scale_loss(y_true, y_pred, self.loss_type, self.loss_scales)
        return tf.reduce_mean(label_similarity)

    def regulariser_loss(self, _, y_pred):
        ndims = len(y_pred.get_shape().as_list()) - 2
        assert ndims in [3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        """ reconstruction loss """
        loss_regulariser = labelloss.local_displacement_energy(y_pred, self.energy_type, self.energy_weight)
        return loss_regulariser
    
class Null_Loss():
    """
    similarity_type:'cross-entropy', 'mean-squared', 'dice', 'doubledice', 'doubledice-ssim', 'jaccard'
    similarity_scales:: [0, 1, 2, 4, 8, 16]
    regulariser_type: 'bending','gradient-l2','gradient-l1'
    """
    def __init__(self, vol_size = (144, 176, 112)):
        self.vol_size = vol_size

    def loss(self, _, y_pred):
        """ reconstruction loss """
        ndims = len(y_pred.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        return y_pred
    
    
class dice_loss_local():
    
    def __init__(self):
        self.count = 0
        print('count =',self.count )

    def loss(self, output, target,loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
        if self.count > 365:
            self.count = 0
        
        if self.count < 200:

            ones = tf.ones([160,192,160])
            output1 = tf.equal(output,ones)
            target1 = tf.equal(target,ones)
            
            output1 = tf.cast(output1,float, name=None)
            target1 = tf.cast(target1,float, name=None)
    
            inse = tf.reduce_sum(output1 * target1, axis=axis)
            if loss_type == 'jaccard':
    
                l = tf.reduce_sum(output1 * output1, axis=axis)
                r = tf.reduce_sum(target1 * target1, axis=axis)
            elif loss_type == 'sorensen':
                l = tf.reduce_sum(output, axis=axis)
                r = tf.reduce_sum(target, axis=axis)
            else:
                raise Exception("Unknow loss_type")
            dice = (2. * inse + smooth) / (l + r + smooth)
            dice = tf.reduce_mean(dice)
            return -dice
        
        if 200 <= self.count < 365:
            
            ones = tf.ones([160,192,160])
            output1 = tf.equal(output,2*ones)
            target1 = tf.equal(target,2*ones)
            
            output1 = tf.cast(output1,float, name=None)
            target1 = tf.cast(target1,float, name=None)
    
            inse = tf.reduce_sum(output1 * target1, axis=axis)
            if loss_type == 'jaccard':
    
                l = tf.reduce_sum(output1 * output1, axis=axis)
                r = tf.reduce_sum(target1 * target1, axis=axis)
            elif loss_type == 'sorensen':
                l = tf.reduce_sum(output, axis=axis)
                r = tf.reduce_sum(target, axis=axis)
            else:
                raise Exception("Unknow loss_type")
            dice = (2. * inse + smooth) / (l + r + smooth)
            dice = tf.reduce_mean(dice)
            return -dice