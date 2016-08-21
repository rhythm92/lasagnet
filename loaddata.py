import gzip     as gz
import cPickle  as pkl
import nibabel  as nii
import random   as rnd 
import numpy    as np
import theano   as t
import matplotlib.pyplot as plt

def shared_type(data, name, dtype=t.config.floatX, borrow=True):
    #return t.shared(np.asarray(data, dtype=dtype), name=name, borrow=borrow)
    return np.asarray(data, dtype=dtype)

def load_mnist(shared=True):
  
  def split_dataset(data_xy, shared=True, borrow=True):
    data_x, data_y = data_xy
    data_x = np.reshape(np.asarray(data_x),(data_x.shape[0],1,28L,28L))
    #return data_x, data_y
    return shared_type(data_x,'X',dtype=t.config.floatX), \
           shared_type(data_y,'Y',dtype=np.int32)
  
  # Load the dataset
  print('Loading Data...')
  f = gz.open('mnist/mnist.pkl.gz','rb')
  train_set, valid_set, tests_set = pkl.load(f)
  print('Done')
  
  return [(split_dataset(tests_set,shared=shared)),
          (split_dataset(valid_set,shared=shared)), 
          (split_dataset(train_set,shared=shared))]
  
def load_flair(shared=True):
  
  def split_tvt(data_x, data_y):
    
    def quarter_split(data):
      q = data.shape[0] // 4
      train = data[     :2*q,:,:]
      valid = data[1+2*q:3*q,:,:]
      tests = data[1+3*q:   ,:,:]
      return train, valid, tests

    def select_and_reshape(data,slice_order):
      data = np.rollaxis(data[:,:,slice_order],2,0)
      return quarter_split(data)
    
    slice_order = np.arange(0,data_x.shape[2])
    rnd.shuffle(slice_order)
    xtrain,xvalid,xtests = select_and_reshape(data_x,slice_order)
    ytrain,yvalid,ytests = select_and_reshape(data_y,slice_order)
    
    return xtrain, ytrain, xvalid, yvalid, xtests, ytests
  
  
  xname = 'D:\IMG\FLAIR\MSSEG16\TrainingSet\min\m16_FLAIR_07.nii.gz'
  yname = 'D:\IMG\FLAIR\MSSEG16\TrainingSet\min\m16_GTC_07.nii.gz'
  rnd.seed(1234)
  
  print('Loading Data...')
  # FLAIR  # vsize = NII.get_header().get_zooms()
  NII   = nii.nifti1.load(xname)
  FLAIR = NII.get_data()
  NII   = nii.nifti1.load(yname)
  GT    = NII.get_data()
  xtrain, ytrain, xvalid, yvalid, xtests, ytests = split_tvt(FLAIR, GT)
  print('Done')
  
  return [(xtrain,ytrain),(xvalid,yvalid),(xtests,ytests)]

  
  
  
  
  
  
   
  