from .gradient import jacobian,hessian
import torch
import functools
import pickle
import os
import numpy as np
def patch_derivate(x,patchnet,device="cpu"):
  x=x.to(device)
  x.requires_grad=True
  y=patchnet(x)
  grad_y=jacobian(y,x)
  x.requires_grad=False
  return grad_y.detach()

def patch_laplace(x,patchnet,device="cpu"):
  x=x.to(device)
  x.requires_grad=True
  y=patchnet(x)
  dim=x.shape[1]
  f=0.0
  for k in range(dim):
    f=f+hessian(y,x,i=k,j=k)
  x.requires_grad=False
  return f.detach()

def patch_rhs(x,patchnet,beta,device="cpu"):  
  # beta is a function
  x=x.to(device)
  x.requires_grad=True
  y=patchnet(x)
  f=0
  # breakpoint()
  product=beta(x)[:,None]*jacobian(y,x)
  for k in range(x.shape[1]):
    f+=jacobian(product,x,i=k,j=k)
  x.requires_grad=False
  return f.detach()

def cal_derivative(x,phi):
    origin_mod=x.requires_grad
    if origin_mod==False:
      x.requires_grad=True
    try:
      y=jacobian(phi(x).reshape(-1,1),x)
    except Exception:  # phi(x) does not have a grad_fn, means that grad is zero
      y=torch.zeros_like(x)
    if origin_mod==False:
      x.requires_grad=False
    return y.detach()

def pickle_load(path):
  assert os.path.isfile(path)
  with open(path, 'rb') as f:
    return pickle.load(f)

def pickle_save(data,path):
  with open(path,"wb") as f:
    pickle.dump(data,f)


def params_pack(vector, signature):
  # Transform the flattened parameters (a vector) into theirs original form `dict(named_parameters)`
  # given a signature.
  params={}
  cnt=0
  for key,val in signature.items():
    mm=list(val)  #  shape of a tensor
    num=int(functools.reduce(lambda x,y:x*y,mm,1))
    params[key]=torch.nn.Parameter(vector[cnt:cnt+num].reshape(val))
    cnt+=num
  return params

def params_unpack(params):
  cnt = 0
  signature={}
  for name,param in params.items(): 
      signature[name]=param.shape
      param = param.detach()    # important step so that vector.requires_grad is False
      vector = param.view(-1) if cnt == 0 else torch.hstack([vector, param.view(-1)])
      cnt = 1
  return vector, signature

def seed_all(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
  np.random.seed(seed)  # Numpy module.


def params_to_device(params, device):
  for key,val in params.items():
    params[key]=val.to(device)

import gc
from .gradient import caches_clear
def clear_cuda_cache():
    caches_clear()
    gc.collect()
    torch.cuda.empty_cache()