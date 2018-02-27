pass
from stats232a.layers import *


def fc_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    out, cache = None, None
    
    ###########################################################################
    # TODO: Implement fc-relu forward pass.                                   #
    ###########################################################################
    a, fc_cache = fc_forward(x,w,b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return out, cache


def fc_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    dx, dw, db = None, None, None
    
    ###########################################################################
    # TODO: Implement the fc-relu backward pass.                              #
    ###########################################################################
    fc_cache, relu_cache = cache
    da =relu_backward(dout, relu_cache)
    dx,dw,db = fc_backward(da, fc_cache)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return dx, dw, db



