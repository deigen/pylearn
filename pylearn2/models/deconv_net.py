from pylearn2.models.model import Model
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import FixedVarDescr
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX
import theano.tensor as T

import theano, theano.tensor as T
from theano.tensor.nnet import conv

import numpy as np

theano.config.compute_test_value = 'raise'

def _pos_shrink(x, t):
    return T.maximum(0, x - t)

class DeconvNet(Model):
    """
    A deconvolutional network.
    TODO add paper reference
    """

    def __init__(self, batch_size, input_shape, input_channels):

        self.__dict__.update(locals())
        del self.self

        # hardcoded params TODO
        self.shrink_thresh = 0.5
        self.ista_rate = 0.005
        self.ista_iters = 1
        self.filter_shape = (8,3,5,5)
        (hid_channels, fc, fi, fj) = self.filter_shape
        assert fc == input_channels

        (vi, vj) = input_shape
        hidi = vi - fi + 1
        hidj = vj - fj + 1
        hid_shape = [hidi, hidj]

        self.input_space = Conv2DSpace(input_shape, input_channels)
        self.output_space = Conv2DSpace(hid_shape, hid_channels,
                                        axes=('b', 'c', 0, 1))

        self.force_batch_size = batch_size
        self._test_batch_size = batch_size

        self._params = []

        init_W = np.random.randn(*self.filter_shape)
        Wnorm = np.sqrt(np.sum(init_W.reshape((hid_channels, -1))**2, axis=1))
        init_W /= Wnorm.reshape((hid_channels, 1, 1, 1))

        self.W = sharedX(value=init_W, name='W')
        self._params.append(self.W)

        self.W_t = self.W.transpose((1,0,2,3))[:,:,::-1,::-1]
        self.filter_shape_t = (input_channels, hid_channels, fi, fj)

    def reconstruct(self, z):
        '''Computes the reconstructed input given the the hidden layer'''
        return conv.conv2d(z, self.W_t, border_mode='full')

    def ista_iter(self, x, z):
        '''computes a single ista iteration'''
        vc = self.input_space.nchannels
        hc = self.output_space.nchannels
        vis_shape = [self.batch_size, vc] + self.input_space.shape
        hid_shape = [self.batch_size, hc] + self.output_space.shape

        rate = self.ista_rate
        shrink_thresh = self.shrink_thresh

        Wz = conv.conv2d(z, self.W_t,
                         image_shape=hid_shape,
                         filter_shape=self.filter_shape_t,
                         border_mode='full')
        
        x = x.transpose((0,3,1,2))
        Wd = conv.conv2d(Wz - x, self.W,
                         image_shape=vis_shape,
                         filter_shape=self.filter_shape,
                         border_mode='valid')

        return _pos_shrink(z - rate*Wd, rate*shrink_thresh)

    def censor_updates(self, updates):
        """
        Modify the updates proposed by the training algorithm to
        preserve the norm constraint on the kernels.
        """

        if self.W in updates:
            W = updates[self.W]
            norms = T.sqrt(T.sqr(W).sum(axis=(1,2,3)))
            W = W / (1e-7 + norms.reshape((norms.shape[0], 1, 1, 1)))
            updates[self.W] = W

class InferenceCallback(object):

    def __init__(self, model, code):
        self.__dict__.update(locals())
        del self.self

        (xi, xj) = model.input_shape
        xc = model.input_space.nchannels
        self.x_buf = sharedX(np.zeros((model.batch_size, xi, xj, xc)))

        # init buffers func
        xv = T.tensor4('xv')
        xv.tag.test_value = self.x_buf.get_value()
        self.do_init = theano.function([xv], (),
                                  updates={self.x_buf: xv,
                                           code: T.zeros_like(code)})

        # single ista iter func
        self.do_ista_iter = theano.function((), (), updates=
              {code: self.model.ista_iter(self.x_buf, code)})


    def __call__(self, X, Y):
        """
        updates self.code

        X: a numpy tensor for the input image of shape (batch_size, rows, cols, channels)
        Y: unused

        the model is available as self.model
        """

        self.do_init(X)
        for it in xrange(self.model.ista_iters):
            self.do_ista_iter()

class DeconvNetMSESparsity(Cost):
    """
    The standard cost for training a deconvolution network.

    """

    def __call__(self, model, X, Y=None, deconv_net_code=None, **kwargs):
        """
            Returns a theano expression for the mean squared error.

            model: a DeconvNet instance
            X: a theano tensor of shape (batch_size, rows, cols, channels)
            Y: unused
            deconv_net_code: the theano shared variable representing the deconv net's code
            kwargs: unused
        """

        # Training algorithm should always supply the code
        assert deconv_net_code is not None

        recons = model.reconstruct(deconv_net_code)
        cost = T.sum((recons - X.transpose((0,3,1,2)))**2)
        return cost

    def get_fixed_var_descr(self, model, X, Y):
        """
            Returns a FixedVarDescr describing how to update the code.
            We use this mechanism because it is the easiest way to use a python
            loop to do inference.

            model: a DeconvNet instance
            X: a theano tensor of shape (batch_size, rows, cols, channels)
            Y: unused
        """

        rval = FixedVarDescr()

        code = sharedX(model.output_space.get_origin_batch(model.batch_size))

        rval.fixed_vars = {'deconv_net_code' : code}

        rval.on_load_batch = [InferenceCallback(model, code)]

        return rval
