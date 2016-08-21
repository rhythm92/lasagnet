import numpy as np

def cat(*args):
  x = ()
  for a in args:
    if type(a) is tuple:
      x+=a
    elif type(a) in {list,np.ndarray}:
      x+=tuple(a)
    else:
      x+=(a,)
  return tuple(x)
