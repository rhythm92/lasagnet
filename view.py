import matplotlib.pyplot as plt
import numpy             as np
import time
import lasagne           as l
import lasagne.layers    as ll

def show_filters(net, layernum=1):
  
  def grid(N):
    nx = np.ceil(np.sqrt(N))
    ny = np.floor(N/nx)
    return nx, ny

  # get the data
  knettle = ll.get_all_layers(net)
  Wb = knettle[layernum].get_params()
  W  = Wb[0].get_value()
  
  # subplot layout
  nx, ny = grid(W.shape[0])
  for iy in np.arange(0,ny):
    for ix in np.arange(0,nx):
      i  = (1 + ix + (iy)*nx).astype(np.int32)
      ax = plt.subplot(ny,nx,i)
      ax.imshow(np.squeeze(W[i-1,0,:,:]))
      ax.axis('off')
  plt.show(block=False)
  plt.draw()
  plt.pause(0.01)

def show_slice(volume, z):
  print(volume.shape)
  plt.imshow(np.squeeze(volume[z,:,:]))
  plt.axis('off')
  plt.show(block=False)
  plt.draw()
  plt.pause(0.01)
  
  
  
  