import numpy as np
import torch
import itertools
from .geometry import Geometry
from .sampler import random_bbox_sampler,uniform_bbox_sampler,sample
from ..gradient import jacobian 
import os 
class FunctionBasedGeometry(Geometry):
  """
  a geometry area based on a level set function theta by choosing func(x)<0 as the area.
  """
  def __init__(self,dim, bbox, diam, targetfunc,parametric_func=None,parameters_num=None,targetfunc_=None):
    """
    bbox: [xmin,xmax]
    
    targetfunc: defining the target area by targetfunc(x)<0.
          targetfunc have to support numpy.ndarray of shape (n,dim) as inputs.
    paramertic_func: parametric form of boundery:
            if None then use a random appoximate approach to get boundery from targetfunc
    parameters_num: it's used only if parametric_func is provided

    targetfunc_: torch-based implement of targetfunc, which helps to find boundary_normal
    warning:  must to choose bbox larger enough to cover the whole area defined by targetfunc 
    """
    super().__init__(dim, bbox, diam)
    self.targetfunc=targetfunc
    self.parametric_func=parametric_func
    self.parameters_num=parameters_num
    self.targetfunc_=targetfunc_
    
  def inside(self,x):
    return (self.targetfunc(x)<=0+1e-10)
    # return (self.targetfunc(x)<=0).flatten()
  
  def on_boundary(self,x, atol=1e-8):
    return np.isclose(self.targetfunc(x),np.zeros((x.shape[0],)),atol=atol)

  def boundary_normal(self, x):
      """Compute the unit normal at x for Neumann or Robin boundary conditions.
      inputs np.ndarray and outputs np.ndarray
      """
      if not self.targetfunc_: 
        raise Exception(
            "{}.boundary_normal couldn't be implemented without torch-based targetfunc_".format(self.idstr)
        )

      x=torch.tensor(x)
      x.requires_grad=True
      y=self.targetfunc_(x)[:,None]
      dx=jacobian(y,x)
      dx=dx/torch.norm(dx,dim=1,keepdim=True)
      return dx.detach().numpy()



  def uniform_points(self,n):
    """
    Firstly estimate area ratio of geo over its bbox
    Then uniform sampling points inside bbox, and filter it by targetfunc(x)<0
    """
    temp=self.inside(random_bbox_sampler(n,self.dim,self.bbox))
    count=len(temp[temp==True])      #ratio:  `count` over n
    temp=uniform_bbox_sampler(int(n*n/count+1),self.dim,self.bbox)
    return temp[self.inside(temp)].astype(np.float64)
  
  def uniform_boundary_points(self, n):
    """
    if paramertic_func is provided,we can easily sample the boundary points uniformly in the parametric space.
    otherwise it is not supported.
    """    
    if self.parametric_func:
      if self.parameters_num==1:
        return self.parametric_func(np.linspace(0,1,n)[:,None].astype(np.float64))
      else:
        n_base=int(np.ceil(n**(1/self.parameters_num)))
        pool=[]
        for i in range(self.parameters_num):
          pool.append(np.linspace(0,1,n_base))
        return self.parametric_func(np.array(list(itertools.product(*pool))).astype(np.float64))
         
    else:
        print("Warning: {}.uniform_boundary_points not implemented because parametric_func for boundary hasn't been provided. Use random_boundary_points instead.".format(self.idstr))
        return self.random_boundary_points(n)
  def random_boundary_points(self,max_n, random="pseudo"):
    """
    either from paramertic_func or a random appoximate of boundery
    if paramertic_func is provided, then each input must be parameter in [0,1] 
    otherwise, we will use the following approach to get boundary points:
      1.sample n initial points in bbox. 
      2.adjust each point to its nearest boundary
        2.1 using SGD or Adam in finite steps
        2.2 select points that close enough to boundary 
    
    """
    if self.parametric_func:
      return self.parametric_func(sample(max_n,self.parameters_num,random).astype(np.float64))
    elif self.targetfunc_:
      step=500;learning_rate=0.001
      x=torch.tensor(random_bbox_sampler(max_n,self.dim,self.bbox,random)).to(torch.float64)
      x = x[self.targetfunc_(x)< 0.1 and self.targetfunc_(x)> -0.1].clone()
      x.requires_grad=True
      opt=torch.optim.Adam([x],lr=learning_rate)
      for i in range(step):
        loss=torch.sum(self.targetfunc_(x)**2)/max_n
        loss.backward()
        opt.step()
        opt.zero_grad()
      x=x.detach().cpu().numpy()
      x=x[np.isclose(self.targetfunc(x),np.zeros((max_n,)))]
      return x
    else:
      raise Exception("neither given a paramatric function, nor a torch-based targetfunc_")
      
  def random_points(self,n,random="pseudo"):
    x = np.empty(shape=(n, self.dim))
    i = 0
    while i < n:
        tmp = random_bbox_sampler(n,self.dim,self.bbox,random)
        tmp = tmp[self.inside(tmp)]
        if len(tmp) > n - i:
            tmp = tmp[: n - i]
        x[i : i + len(tmp)] = tmp
        i += len(tmp)
    return x.astype(np.float64)
  
class Ellipsoid(FunctionBasedGeometry):
  # a standard 3d ellipsoid geometry 
  def __init__(self, center, semiaxes):
    dim=3
    self.center= np.array(center)
    self.semiaxes= np.array(semiaxes)
    assert np.all(self.semiaxes>0)
    self.min_semiaxes=min(semiaxes)
    bbox = np.vstack([self.center-self.semiaxes,self.center+self.semiaxes])
    diam = 2 * np.max(self.semiaxes)
    targetfunc = lambda x: (x[:,0]/self.semiaxes[0])**2 + (x[:,1]/self.semiaxes[1])**2 + (x[:,2]/self.semiaxes[2])**2 -1
    super().__init__(dim, bbox, diam, targetfunc, targetfunc_=targetfunc)
    self.min_semiaxes=min(semiaxes)
    self.seed = None  # the seed for matlab.engine random number generation
    

  def random_boundary_points(self,max_n):
    # the implementation is based on https://www.zhihu.com/question/453759946.
    # randomly sample points in a unit sphere, project it to ellipsoid and accept it with a probability
    
    X = np.random.normal(size=(max_n, self.dim)).astype(np.float64)
    X = X/np.linalg.norm(X,axis=1,keepdims=True)
    X=self.semiaxes*X
    P=np.sqrt(X[:,0]**2/self.semiaxes[0]**4+X[:,1]**2/self.semiaxes[1]**4+X[:,2]**2/self.semiaxes[2]**4)* self.min_semiaxes
    idx= np.random.rand(max_n)<P
    X=X[idx]
    print(f"{len(X)} points sampled on the ellipsoid surface")
    return X  
  def uniform_boundary_points(self, n=None, h=None):
    # call distmesh in python
    #  n : numbers estimated to sampled
    #  h : use as the guess length of mesh edges
    #  use `h` if provided, otherwise use `n` to estimate `h`. 
    from .distmesh import distmeshnd, bndproj
    assert n or h
    center=self.center
    sa= self.semiaxes
    if h == None:
      area= 4*np.pi*(sa[0]*sa[1]+sa[1]*sa[2]+sa[0]*sa[2])/3
      h=np.sqrt(area/(n*1.3)) # Estimated edge length
    # fd= eng.eval(f"@(p) ((p(:,1)-{center[0]})/{sa[0]}).^2+((p(:,2)-{center[1]})/{sa[1]}).^2+\
    #               ((p(:,3)-{center[2]})/{sa[2]}).^2- 1")
    # fh= eng.eval("@(p) ones(size(p,1),1)")
    fd = lambda p: ((p[:,0]-center[0])/sa[0])**2+((p[:,1]-center[1])/sa[1])**2+((p[:,2]-center[2])/sa[2])**2-1
    fh = lambda p: np.ones(p.shape[0])
    bbox= np.double(self.bbox)
    p,t =distmeshnd(fd,fh,h,bbox,fig=None)
    for i in range(5):
      bndproj(p,t,fd)
    p=p[self.on_boundary(p,atol=1e-7)].copy()
    print(f"{len(p)} points sampled on the ellipsoid surface")
    return p
  
  # # old matlab version: not recommended.
  # def uniform_boundary_points(self, n=None, h=None):
  #   # call distmesh.m using  matlab.engine or octave 's oct2py
  #   # Instructions please follow https://stackoverflow.com/questions/51406331/how-to-run-matlab-code-from-within-python
  #   #  n : numbers estimated to sampled
  #   #  h : use as the guess length of mesh edges
  #   #  use `h` if provided, otherwise use `n` to estimate `h`. 
  #   try:
  #     import matlab
  #     import matlab.engine
  #     assert n or h
  #     center=self.center
  #     sa= self.semiaxes
  #     if h == None:
  #       area= 4*np.pi*(sa[0]*sa[1]+sa[1]*sa[2]+sa[0]*sa[2])/3
  #       h=np.sqrt(area/(n*1.3)) # Estimated edge length
  #     eng = matlab.engine.start_matlab()
  #     if type(self.seed)==int:
  #       eng.eval(f"rng({self.seed})")
  #     eng.cd(os.path.join(os.path.dirname(__file__),"distmesh"))
  #     fd= eng.eval(f"@(p) ((p(:,1)-{center[0]})/{sa[0]}).^2+((p(:,2)-{center[1]})/{sa[1]}).^2+\
  #                  ((p(:,3)-{center[2]})/{sa[2]}).^2- 1")
  #     fh= eng.eval("@(p) ones(size(p,1),1)")
  #     bbox= matlab.double(np.double(self.bbox))
  #     p,t =eng.distmesh(fd,fh,h,bbox,nargout=2)
  #     eng.exit()
  #     p=np.array(p)
  #     p=p[self.on_boundary(p,atol=1e-7)]
  #     print(f"{len(p)} points sampled on the ellipsoid surface")
  #     return p
  #   except ImportError:
  #     pass
  #   print("Warning: {}.uniform_boundary_points is supported only after matlab.engine is installed. Use random_boundary_points instead.".format(self.idstr))
  #   return self.random_boundary_points(n)

class GaussianMixture(FunctionBasedGeometry):
  def __init__(self, dim, bbox, diam, means, covs, weights, lowerbound):
    num = len(means)
    self.means= np.array(means)  #(m, d)
    self.covs= np.array(covs)     #(m, d, d)
    self.weights= np.array(weights) # (m,)
    self.lowerbound = lowerbound    # (,)
    assert self.covs.shape[1]==dim
    assert self.means.shape[1]==dim
    assert num == self.weights.shape[0]
    assert num == self.covs.shape[0]
    det_cov = np.linalg.det(self.covs)  #(m,)
    assert np.all(det_cov>0)
    inv_cov = np.linalg.inv(self.covs)   #(m, d, d)

    def pdf_np(x: np.ndarray):  # x : (n,d)
      assert len(x.shape)==2
      x = np.expand_dims(x, axis=-2)  # (n,1,d)
      x_minus_mu = (x - self.means)[...,None]    # (n,m,d,1)
      x_minus_mu_T = x_minus_mu.transpose(0,1,3,2)  # (n,m,1,d)
      exponent= np.exp(-0.5*np.matmul(np.matmul(x_minus_mu_T, inv_cov),x_minus_mu).squeeze(-1).squeeze(-1))  # (n,m)
      ps = exponent /np.sqrt(det_cov)/(2*np.pi)**(dim/2)   # (n,m)
      # if len(ps.shape)!=2:
      #   breakpoint()
      return np.sum(self.weights*ps,axis=1)  # (n,)
    
    def pdf_torch(x: torch.Tensor):
      x = x.unsqueeze(-2)  # (n,1,d)
      x_minus_mu = (x - torch.from_numpy(self.means))[...,None]    # (n,m,d,1)
      x_minus_mu_T = x_minus_mu.permute(0,1,3,2)  # (n,m,1,d)
      exponent= torch.exp(-0.5*torch.matmul(torch.matmul(x_minus_mu_T, torch.from_numpy(inv_cov)),x_minus_mu).squeeze())  # (n,m)
      x = exponent /torch.sqrt(torch.from_numpy(det_cov))/(2*torch.pi)**(dim/2)   # (n,m)
      return torch.sum(torch.from_numpy(self.weights)*x,dim=1)  # (n,)
    targetfunc = lambda x: lowerbound - pdf_np(x)
    targetfunc_ = lambda x:  lowerbound - pdf_torch(x)
    super().__init__(dim, bbox, diam, targetfunc, targetfunc_=targetfunc_)
    

  def uniform_boundary_points(self, n=None, h=None):
    # call distmesh in python
    #  n : numbers estimated to sampled
    #  h : use as the guess length of mesh edges
    #  use `h` if provided, otherwise use `n` to estimate `h`. 
    from .distmesh import distmesh2d, distmeshnd, bndproj
    assert n or h
    fd = self.targetfunc
    fh = lambda p: np.ones(p.shape[0])
    bbox= np.double(self.bbox).flatten()
    if self.dim==2:
      if n is None:
        p,t =distmesh2d(fd,fh,h,bbox)
      else:
        h = 0.1
        p,t =distmesh2d(fd,fh,h,bbox,fig=None)
        for i in range(5):
          bndproj(p,t,fd)
        h2 = np.sum(self.on_boundary(p,atol=1e-7))*h/n # estimate the disired h by estimate the perimeter first.
        p,t =distmesh2d(fd,fh,h2,bbox,fig=None)
    else:
      p,t =distmeshnd(fd,fh,h,bbox, fig =None)
    for i in range(5):
      bndproj(p,t,fd)
    p=p[self.on_boundary(p,atol=1e-7)].copy()
    print(f"{len(p)} points sampled on the gaussian distrtion boundary ")
    return p

      
      
      
    

        