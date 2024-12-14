import torch
from .gradient import jacobian,hessian
from . import geometry as geo
import numpy as np
from .utils import patch_derivate,patch_laplace,patch_rhs,cal_derivative
class ReferenceU:
  """
  torch-based implement, for autograd usage

  here assumpe beta1 and beta2 are piecewise constants.
  """
  def __init__(self,g,u1,u2,beta1=1,beta2=1):
    self.g=g  # DIY geometry
    self.u1=u1
    self.u2=u2
    self.beta1=beta1
    self.beta2=beta2
  def u_whole(self,x):
    y=torch.empty([x.shape[0],1]).to(x)
    index=self.g.inside_u1(x.detach().cpu().numpy())
    
    y[index]=self.u1(x[index])[:,None]
    y[~index]=self.u2(x[~index])[:,None]
    return y

  def f(self,x):
    """
    """
    x.requires_grad=True
    total=x.shape[0]
    y=torch.zeros([total,1])
    inner_index=self.g.inside_u1(x.detach().cpu().numpy())
    
    inner=self.u1(x[inner_index])[:,None]
    outer=self.u2(x[~inner_index])[:,None]
    for k in range(x.shape[1]):
      y+=-hessian(inner,x,i=k,j=k)*self.beta1
      y+=-hessian(outer,x,i=k,j=k)*self.beta2
    x.requires_grad=False
    return y.detach()
  
  def f1_(self,x):
    x.requires_grad=True
    y=self.u1(x)[:,None]
    y1=torch.zeros_like(y)
    for k in range(x.shape[1]):
      y1+=-hessian(y,x,i=k,j=k)*self.beta1
    x.requires_grad=False
    return y1.detach()

  def f2_(self,x):
    x.requires_grad=True
    y=self.u2(x)[:,None]
    y1=torch.zeros_like(y)
    for k in range(x.shape[1]):
      y1+=-hessian(y,x,i=k,j=k)*self.beta2
    x.requires_grad=False
    return y1.detach()


  def w(self,x):
    return (self.u2(x)-self.u1(x))[:,None]
    
  def g_D(self,x):
    return self.u2(x)[:,None].detach()
  
  def v(self,x):
    """
    maintain requires_grad implemention
    """
    origin_mod=x.requires_grad
    if origin_mod==False:
      x.requires_grad=True
    y=torch.sum(jacobian((self.beta2*self.u2(x)-self.beta1*self.u1(x)).reshape(-1,1),x)* \
        torch.tensor(self.g.interface_normal_vector(x.detach().cpu().numpy())),dim=1,keepdim=True)
    if origin_mod==False:
      x.requires_grad=False
    return y.detach()

class ReferenceU_vc:
  """
  torch-based implement, for autograd usage

  here assumpe beta1 and beta2 are Variable-coefficient
  """
  def __init__(self,g,u1,u2,beta1=None,beta2=None):
    self.g=g  # DIY geometry
    self.u1=u1
    self.u2=u2
    if beta1==None:
      self.beta1 =lambda x: 1.0*torch.ones(len(x))
    else:
      self.beta1=beta1
    if beta2==None:
      self.beta2 =lambda x: 1.0*torch.ones(len(x))
    else:
      self.beta2=beta2
  
  # def u1(self,x):
  #   return torch.exp(x[:,0]**2+x[:,1]**2)
  # def u2(self,x):
  #   return 0.1*((x[:,0]**2+x[:,1]**2)**2-0.01*torch.log(2 *torch.sqrt(x[:,0]**2+x[:,1]**2)))

  def u_whole(self,x):
    y=torch.empty([x.shape[0],1])
    index=self.g.inside_u1(x.detach().cpu().numpy())
    
    y[index]=self.u1(x[index])[:,None]
    y[~index]=self.u2(x[~index])[:,None]
    return y

  def f(self,x):
    """
    """
    x.requires_grad=True
    total=x.shape[0]
    y=torch.zeros([total,1])
    inner_index=self.g.inside_u1(x.detach().cpu().numpy())
    
    inner=self.u1(x[inner_index])[:,None]
    outer=self.u2(x[~inner_index])[:,None]
    inner_product=(self.beta1(x)[:,None]*jacobian(inner,x))[inner_index]
    outer_product=(self.beta2(x)[:,None]*jacobian(outer,x))[~inner_index]
    for k in range(x.shape[1]):
      y+=-jacobian(inner_product,x,i=k,j=k)
      y+=-jacobian(outer_product,x,i=k,j=k)
    x.requires_grad=False
    return y.detach()
  
  def f1_(self,x):
    x.requires_grad=True
    y=self.u1(x)[:,None]
    y1=torch.zeros_like(y)
    product=self.beta1(x)[:,None]*jacobian(y,x)
    for k in range(x.shape[1]):
      y1+=-jacobian(product,x,i=k,j=k)
    x.requires_grad=False
    return y1.detach()

  def f2_(self,x):
    x.requires_grad=True
    y=self.u2(x)[:,None]
    y1=torch.zeros_like(y)
    product=self.beta2(x)[:,None]*jacobian(y,x)
    for k in range(x.shape[1]):
      y1+=-jacobian(product,x,i=k,j=k)
    x.requires_grad=False
    return y1.detach()


  def w(self,x):
    return (self.u2(x)-self.u1(x))[:,None].detach()
    
  def g_D(self,x):
    return self.u2(x)[:,None].detach()
  
  def v(self,x):
    """
    maintain requires_grad implemention
    """
    origin_mod=x.requires_grad
    if origin_mod==False:
      x.requires_grad=True
    beta1=self.beta1(x).detach()
    beta2=self.beta2(x).detach()
    y=torch.sum(jacobian((beta2*self.u2(x)-beta1*self.u1(x)).reshape(-1,1),x)* \
        torch.tensor(self.g.interface_normal_vector(x.detach().cpu().numpy())),dim=1,keepdim=True)
    if origin_mod==False:
      x.requires_grad=False
    return y.detach()

class CompoundGeometry:
  def __init__(self,geoIn,geoOut):
    """
    geoIn: inner geomerty
    geoOut: outer geomerty, such that inner_domain=geoIn and outer_domain=geoOut-geoIn

    """
    self.dim=geoOut.dim
    self.bbox=geoOut.bbox    #np.array([[-1,-1],[1,1]])
    self.diam=geoOut.diam

    self.subdomain1=geoIn
    self.whole_domain=geoOut
    self.subdomain2=geoOut-geoIn

  def inside(self,x):
    return self.whole_domain.inside(x)

  def inside_u1(self,x):
    return self.subdomain1.inside(x)

  def inside_u2(self,x):
    return self.subdomain2.inside(x)
  
  def on_boundary(self,x):
    return self.whole_domain.on_boundary(x)

  def interface_normal_vector(self,x):
    return self.subdomain1.boundary_normal(x)


  def interface_sampler(self,max_n,random=None, h=None):
    #  `random` shall be one of the [None, "pseudo", "LHS", "Halton", "Hammersley", "Sobol"]
    # if `random`=None, a uniform sampler is performed
    if random:
      return self.subdomain1.random_boundary_points(max_n,random=random)
    else:
      if h:
        return self.subdomain1.uniform_boundary_points(h=h)
      else:
        return self.subdomain1.uniform_boundary_points(max_n)

  def boundary_sampler(self,max_n, random=None):
    if random:
      return self.whole_domain.random_boundary_points(max_n,random=random)
    else:
      return self.whole_domain.uniform_boundary_points(max_n)

  def whole_sampler(self,n, random=None, min_sample_sub1=None):
    if random:
      points = self.whole_domain.random_points(n,random=random)
    else:
      points = self.whole_domain.uniform_points(n)
    if min_sample_sub1 is not None:
      count1= np.sum(self.subdomain1.inside(points))
      if count1< min_sample_sub1:
        addition_points = self.subdomain1.random_points(min_sample_sub1, random=random)
        points = np.concatenate([points[~self.subdomain1.inside(points)], addition_points],axis=0)
    return points

  def uniform_sampler(self,n):
    """
    returns: [whole_point,inner_point,outerpoint]
    """
    whole_point=self.whole_domain.uniform_points(n)
    inner_point=whole_point.copy()[self.inside_u1(whole_point)]
    outer_point=whole_point.copy()[self.inside_u2(whole_point)]

    return whole_point,inner_point,outer_point


# for constants or piecewise constants coefficients

def databuilderforPatch(g_co,reference_u,n_interface=100,device="cpu", h_interface=None):
  """
  last_x_dict: if None, collect new data, otherwise it will append new data to old data
  """
  x_dict={}
  data_dict={}
  interface=torch.tensor(g_co.interface_sampler(n_interface, h=h_interface)).to(torch.float64)
  

  w=reference_u.w(interface)
  v=reference_u.v(interface)
  interface_normal_vector=torch.tensor(g_co.interface_normal_vector(interface.numpy())).to(torch.float64)
  f_jump=(reference_u.f2_(interface)-reference_u.f1_(interface))[:,None].detach()

  x_dict["interface"]=interface.requires_grad_().to(device)

  data_dict["interface"]={"w":-w.to(device),"v":-v.to(device),
                          "interface_normal_vector":interface_normal_vector.to(device),
                          "f_jump":f_jump.to(device),
                          }

  return x_dict,data_dict


def databuilderforBase(g_base, patchnet, reference_u, random_method=None , n_inner=900, n_boundary=100, n_interface=120, device="cpu"):
  """
    `random` shall be one of the [None, "pseudo", "LHS", "Halton", "Hammersley", "Sobol"]
            if `random`=None, a grid sampler is performed
  """
  beta1=reference_u.beta1  # piecewise constants coefficients
  beta2=reference_u.beta2
  x_dict={}
  data_dict={}
  
  inner=torch.tensor(g_base.whole_sampler(n_inner,random=random_method)).to(torch.float64)
  boundary=torch.tensor(g_base.boundary_sampler(n_boundary)).to(torch.float64)
  interface=torch.tensor(g_base.interface_sampler(n_interface)).to(torch.float64)

  f=reference_u.f(inner)
  patch_index=g_base.inside_u1(inner.numpy())
  f[patch_index]+=beta1*patch_laplace(inner[patch_index],patchnet,device).to("cpu")
  sign=torch.ones_like(f)
  sign[~patch_index]=-1

  g_D=reference_u.g_D(boundary)

  v=reference_u.v(interface).to(device)
  interface_normal_vector=torch.tensor(g_base.interface_normal_vector(interface.numpy())).to(torch.float64).to(device)
  normal_jump=beta1*torch.sum(patch_derivate(interface,patchnet,device)*interface_normal_vector, dim=1, keepdim=True) + v
  
  beta=torch.ones_like(f)*beta1
  beta[~patch_index]=beta2 


  x_dict["inner"]=inner.requires_grad_().to(device)
  x_dict["boundary"]=boundary.requires_grad_().to(device)
  x_dict["interface"]=interface.requires_grad_().to(device)

  data_dict["inner"]={"f":f.to(device), "sign":sign.to(device), "beta": beta.to(device)}
  data_dict["boundary"]={"g_D":g_D.to(device)}
  data_dict["interface"]={"normal_jump":normal_jump.to(device),"normal_vector":interface_normal_vector}

  return x_dict,data_dict

# for variable coefficients (continuous or discontinuous)

def databuilderforPatchVC(g_co,reference_u,n_interface=100,device="cpu", h_interface=None):
  """
  For variable coefficents case where coefficents are functions
  
  """
  x_dict={}
  data_dict={}
  interface=torch.tensor(g_co.interface_sampler(n_interface,h=h_interface)).to(torch.float64)
  

  w=reference_u.w(interface)
  v=reference_u.v(interface)
  interface_normal_vector=torch.tensor(g_co.interface_normal_vector(interface.numpy())).to(torch.float64)
  f_jump=(reference_u.f2_(interface)-reference_u.f1_(interface))[:,None].detach()

  ibeta1=reference_u.beta1(interface)[:,None].detach()
  igradbeta1=cal_derivative(interface,reference_u.beta1)

  x_dict["interface"]=interface.requires_grad_().to(device)

  data_dict["interface"]={"w":-w.to(device),"v":-v.to(device),
                          "interface_normal_vector":interface_normal_vector.to(device),
                          "f_jump":f_jump.to(device),
                          "ibeta1":ibeta1.to(device),
                          "igradbeta1":igradbeta1.to(device)
                          }

  return x_dict,data_dict

def databuilderforBaseVC(g_base, patchnet, reference_u, random_method=None , n_inner=900, n_boundary=100, n_interface=120, device="cpu", h_interface=None):
  """
    For variable coefficents case where coefficents are functions

    `random` shall be one of the [None, "pseudo", "LHS", "Halton", "Hammersley", "Sobol"]
            if `random`=None, a grid sampler is performed
  """
  beta1=reference_u.beta1  # function
  beta2=reference_u.beta2
  x_dict={}
  data_dict={}
  
  inner=torch.tensor(g_base.whole_sampler(n_inner,random=random_method)).to(torch.float64)
  boundary=torch.tensor(g_base.boundary_sampler(n_boundary)).to(torch.float64)
  interface=torch.tensor(g_base.interface_sampler(n_interface,h=h_interface)).to(torch.float64)

  f=reference_u.f(inner)
  patch_index=g_base.inside_u1(inner.numpy())
  f[patch_index]+=patch_rhs(inner[patch_index],patchnet,beta1,device).to("cpu")
  sign=torch.ones_like(f)
  sign[~patch_index]=-1

  g_D=reference_u.g_D(boundary)

  v=reference_u.v(interface)
  interface_normal_vector=torch.tensor(g_base.interface_normal_vector(interface.numpy())).to(torch.float64)
  normal_jump=beta1(interface)[:,None]*torch.sum(patch_derivate(interface,patchnet,device).detach().cpu()*interface_normal_vector, dim=1, keepdim=True) + v
  
  beta=beta1(inner)[:,None]
  beta[~patch_index]=beta2(inner[~patch_index])[:,None]
  gradbeta=torch.ones(len(inner),inner.shape[1], dtype=torch.float64)
  gradbeta[patch_index]=cal_derivative(inner[patch_index],beta1)
  gradbeta[~patch_index]=cal_derivative(inner[~patch_index],beta2)
  ibeta1=beta1(interface)[:,None]
  ibeta2=beta2(interface)[:,None]

  x_dict["inner"]=inner.requires_grad_().to(device)
  x_dict["boundary"]=boundary.requires_grad_().to(device)
  x_dict["interface"]=interface.requires_grad_().to(device)

  data_dict["inner"]={"f":f.to(device), "sign":sign.to(device), "beta": beta.to(device),"gradbeta":gradbeta.to(device)}
  data_dict["boundary"]={"g_D":g_D.to(device)}
  data_dict["interface"]={"normal_jump":normal_jump.to(device),"normal_vector":interface_normal_vector.to(device),\
                          "ibeta1":ibeta1.to(device),"ibeta2":ibeta2.to(device)}

  return x_dict,data_dict

def databuilderforAll(g_base, reference_u, random_method=None , interface_random=None, n_inner=900, n_boundary=100, n_interface=120, device="cpu", h_interface=None):
  """
  for Basenet and Patchnet together, with variable coefficents
  `random` shall be one of the [None, "pseudo", "LHS", "Halton", "Hammersley", "Sobol"]
            if `random`=None, a grid sampler is performed
  """

  beta1=reference_u.beta1  # function
  beta2=reference_u.beta2
  x_dict={}
  data_dict={}

  inner=torch.tensor(g_base.whole_sampler(n_inner,random=random_method)).to(torch.float64)
  boundary=torch.tensor(g_base.boundary_sampler(n_boundary)).to(torch.float64)
  if interface_random:
    interface=torch.tensor(g_base.interface_sampler(n_interface,random=interface_random)).to(torch.float64)
  else:
    interface=torch.tensor(g_base.interface_sampler(n_interface,h=h_interface)).to(torch.float64)

  f=reference_u.f(inner)
  patch_index=g_base.inside_u1(inner.numpy())
  sign=torch.ones_like(f)
  sign[~patch_index]=0.0

  g_D=reference_u.g_D(boundary) 
  w=reference_u.w(interface)
  v=reference_u.v(interface).to(device)
  interface_normal_vector=torch.tensor(g_base.interface_normal_vector(interface.numpy())).to(torch.float64).to(device)
  beta=beta1(inner)[:,None]
  beta[~patch_index]=beta2(inner[~patch_index])[:,None]
  gradbeta=torch.ones(len(inner),inner.shape[1],dtype=torch.float64)
  gradbeta[patch_index]=cal_derivative(inner[patch_index],beta1)
  gradbeta[~patch_index]=cal_derivative(inner[~patch_index],beta2)
  ibeta1=beta1(interface)[:,None]
  ibeta2=beta2(interface)[:,None]

  x_dict["inner"]=inner.requires_grad_().to(device)
  x_dict["boundary"]=boundary.requires_grad_().to(device)
  x_dict["interface"]=interface.requires_grad_().to(device)

  data_dict["inner"]={"f":f.to(device), "sign":sign.to(device), "beta": beta.to(device),"gradbeta":gradbeta.to(device)}
  data_dict["boundary"]={"g_D":g_D.to(device)}
  data_dict["interface"]={"w":w.to(device),"v":v.to(device),
                          "normal_vector":interface_normal_vector.to(device),\
                          "ibeta1":ibeta1.to(device),"ibeta2":ibeta2.to(device)}

  return x_dict,data_dict 