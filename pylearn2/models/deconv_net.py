from pylearn2.models.model import Model
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import FixedVarDescr
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX

import theano, theano.tensor as T

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

        self.input_space = Conv2DSpace(input_shape, input_channels)
        self.output_space = Conv2DSpace(hid_shape, hid_channels)

        self.params = []

        self.filter_shape = (8,3,5,5)
        (nfilt, fc, fi, fj) = self.filter_shape

        init_W = np.random.randn(*self.filter_shape)
        Wnorm = np.sqrt(np.sum(W.reshape((nfilt, -1))**2, axis=1))
        W /= Wnorm.reshape((nfilt, 1, 1, 1))

        self.W = sharedX(value=W, name='W')
        self.params.append(self.W)

        self.W_t = self.W.transpose((1,0,2,3))[:,:,::-1,::-1]

    def reconstruct(self, z):
        '''Computes the reconstructed input given the the hidden layer'''
        return conv.conv2d(z, self.W_t, border_mode='full')

    def ista_iter(self, x, z, alpha, rate, bsize=None):
        '''computes a single ista iteration'''
        (vc, vi, vj) = self.vis_shape
        (nf, fc, fi, fj) = self.filter_shape
        (nf, hi, hj) = self.hid_shape
        vis_shape = (self.batch_size,) + self.vis_shape
        hid_shape = (self.batch_size,) + self.hid_shape

        Wz = conv.conv2d(z, self.W_t,
                         image_shape=hid_shape,
                         filter_shape=(vc, nf, fi, fj),
                         border_mode='full')
        
        Wd = conv.conv2d(Wz - x, self.W,
                         image_shape=vis_shape,
                         filter_shape=(nf, vc, fi, fj),
                         border_mode='valid')

        return _pos_shrink(z - rate*Wd, rate*alpha)


class InferenceCallback(object):

    def __init__(self, model, code):
        self.__dict__.update(locals())
        del self.self

        self.x = sharedX(np.zeros((model.batch_size,) + self.input_shape))

        # init buffers func
        xv = T.tensor4('xv')
        xv.tag.test_value = self.x.get_value()
        self.do_init = theano.function([xv], (),
                                  updates={x: xv,
                                           code: T.zeros_like(z)})

        # single ista iter func
        self.do_ista_iter = theano.function((), (), updates=
                {code: self.model.ista_iter(x, code, alpha, ista_rate)})


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
        assert code is not None

        recons = model.reconstruct(deconv_net_code)
        cost = T.sum((X - recons)**2)
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
