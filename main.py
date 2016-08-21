import src
import loaddata
import view
import numpy         as np
import theano        as t
import theano.tensor as tt

import lasagne                as l
import lasagne.init           as li
import lasagne.layers         as ll
import lasagne.updates        as lu
import lasagne.objectives     as lo
import lasagne.nonlinearities as lfcn
from matplotlib.pyplot import pause

# ------------------------------------------------------------------------------
# defining the model 

def build_net(input_shape, lr=0.01, mom=0.9, bs=128):
    
  def symbols():
    x = tt.tensor4('X')
    y = tt.ivector('Y')
    return x, y
  
  def define_net(input_shape, x):
    knet = ll.InputLayer    (
                             name         = 'input',
                             shape        = input_shape,
                             input_var    = x
                             )
    knet = ll.Conv2DLayer   (knet,
                             name         = 'conv1',
                             num_filters  = 16,
                             filter_size  = (5,5),
                             stride       = 1, 
                             nonlinearity = lfcn.tanh,
                             W            = li.GlorotUniform()
                             )
    knet = ll.dropout       (knet,
                             name         = 'dropout1', 
                             p            = 0.5
                             )
    knet = ll.DenseLayer    (knet,
                             name         = 'classify',
                             num_units    = np.prod(input_shape),
                             nonlinearity = lfcn.sigmoid
                             )
    return knet
    
  def define_fcns(net, x, y):
    output    = ll.get_output(net)
    test_pred = ll.get_output(net, deterministic=True)
    params    = ll.get_all_params(net, trainable=True)
    loss      = tt.mean(lo.binary_crossentropy(output, tt.flatten(y)),  dtype=t.config.floatX)
    test_loss = tt.mean(lo.binary_crossentropy(test_pred,tt.flatten(y)),dtype=t.config.floatX)
    updates   = lu.nesterov_momentum(loss, params, learning_rate=lr, momentum=mom)
    train_fcn = t.function([x,y], loss, updates=updates, allow_input_downcast=True)
    valid_fcn = t.function([x,y], test_loss,             allow_input_downcast=True)
    
    return output, params, train_fcn, valid_fcn
  
  print('Defining Net...')
  x, y = symbols()
  knet = define_net(input_shape, x)
  output, params, train_fcn, valid_fcn = define_fcns(knet, x, y)
  print('Done')
  
  return knet, output, params, train_fcn, valid_fcn

def train_net(xtrain, ytrain, xvalid, yvalid,
              net, train_fcn, valid_fcn,
              lr=0.01, ne=100, bs=128):
  
  def iterate_batches(x, y, batch_size):
    for i in range(0, len(x)-batch_size+1, batch_size):
      idx = slice(i, i + batch_size)
      yield x[idx], y[idx]
  
  def cmd_update(e, ne, terr, ntb, verr, nvb):
    print("[{}/{}]"     .format(e+1,ne))
    print("  T loss =\t{:.5}".format(terr/ntb))
    print("  V loss =\t{:.5}".format(verr/nvb))
  
  print('Training Net...')
  for epoch in range(ne):
    
    # full batch training
    train_err = 0
    train_batches = 0
    for batch in iterate_batches(xtrain,ytrain,bs):
      xb, yb = batch
      train_err += train_fcn(add_unity_dim(xb,1), yb.flatten())
      train_batches += 1
    
    # full batch validation
    valid_err = 0
    valid_batches = 0
    for batch in iterate_batches(xvalid,yvalid,bs):
      xb, yb = batch
      valid_err += valid_fcn(add_unity_dim(xb,1), yb.flatten())
      valid_batches += 1
    
    # output update
    cmd_update(epoch, ne, train_err, train_batches, valid_err, valid_batches)
    view.show_filters(net)
    
  print('Done')

def define_hypers():
  # should be made .cfg file
  learning_rate = 0.1
  n_epochs      = 10
  batch_size    = 16L
  img_size      = (128L,224L,256L)
  return learning_rate, n_epochs, batch_size, img_size

# ------------------------------------------------------------------------------
# main

def add_unity_dim(x,idx):
  # inserts a unity dimension at index (for tensor compatibility)
  return x.reshape(src.cat(x.shape[0:idx],1L,x.shape[idx:]))

data = loaddata.load_flair()
xtrain = data[0][0]
ytrain = data[0][1]
xvalid = data[1][0]
yvalid = data[1][1]

learning_rate, n_epochs, batch_size, img_size = define_hypers();
input_size = (batch_size,1L,img_size[0],img_size[1])
knet, output, params, train_fcn, valid_fcn = build_net(input_size)
train_net(xtrain, ytrain, xvalid, yvalid, knet, train_fcn, valid_fcn, 
          lr=learning_rate, ne=n_epochs, bs=batch_size)

print("DONE")




