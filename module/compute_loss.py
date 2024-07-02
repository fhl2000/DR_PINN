# from functorch import vmap, grad, jacrev, make_functional
import torch
import sys
from torch.func import vmap,grad,jacrev,functional_call



def compute_loss_bd(func_params, X_bd, U_bd):

    def f(x, func_params):
        output = functional_call(compute_loss_bd._model,func_params, x)
        return output.squeeze(0)
    
    u_pred = f(X_bd, func_params)
    loss_b = u_pred - U_bd
        
    return loss_b.flatten()

## decoupling approach training U 
def compute_loss_f(func_params, X_inner, Rf_inner):
    ### compute -laplace(U)-f in the domain 
    def f(x, func_params):
        output = functional_call(compute_loss_f._model,func_params, x)
        return output.squeeze(0)
    
    grad2_f = (jacrev(grad(f)))(X_inner, func_params)
    dudX2 = (torch.diagonal(grad2_f))
    
    # laplace = (dudX2[0] + dudX2[1])
    laplace = torch.sum(dudX2)
    loss_Res = -laplace - Rf_inner
    return loss_Res.flatten()

def compute_loss_f_inhom(func_params, X_inner, Beta, Rf_inner):
    ### compute -beta*laplace(U)-f in the domain ,where beta is constant
    def f(x, func_params):
        output = functional_call(compute_loss_f_inhom._model,func_params, x)
        return output.squeeze(0)
    grad2_f = (jacrev(grad(f)))(X_inner, func_params)
    dudX2 = (torch.diagonal(grad2_f))
    
    # laplace = (dudX2[0] + dudX2[1])
    laplace = torch.sum(dudX2)
    
    loss_Res = -Beta*laplace - Rf_inner
    return loss_Res.flatten()

def compute_loss_inj(func_params, X_interf, U_inj):
    # derivative normal jump on the interface for training U (rawnet)
    def f(x,z,func_params):
        x=torch.hstack([x,z])
        output = functional_call(compute_loss_inj._model,func_params,x)
        return output.squeeze(0)
    phi=compute_loss_inj._phi
    z=phi(X_interf)
    loss_inj =  2 * grad(f,argnums=1)(X_interf,z,func_params)* torch.norm(grad(phi)(X_interf))- U_inj
    return loss_inj.flatten()

# def compute_loss_inj2(func_params, X_interf, U_inj, Normal_ij):
#     # derivative normal jump on the interface for training U (rawnet), inhomogeneous cases( piecewise constant beta)
#     def f(x,z,func_params):
#         x=torch.hstack([x,z])
#         output = functional_call(compute_loss_inj2._model, func_params, x)
#         return output.squeeze(0)
#     phi=compute_loss_inj2._phi
#     beta1=compute_loss_inj2._beta1
#     beta2=compute_loss_inj2._beta2
#     z=torch.tensor(0.0,dtype=torch.float64,device=X_interf.device)
#     df=grad(lambda x:f(x,phi(x),func_params))(X_interf)
#     loss_inj =  (beta2-beta1)*torch.sum(df*Normal_ij)+ \
#           2 * beta1* grad(f,argnums=1)(X_interf,z,func_params)* torch.norm(grad(phi)(X_interf))- U_inj
#     return loss_inj.flatten()

def compute_loss_f_vc(func_params, X_ij, Rf_inner, beta, gradbeta):
    ### compute -div(beta*div(U))-f in the domain, beta(x) is the varible coefficient
    def f(x, func_params):
        output = functional_call(compute_loss_f_vc._model,func_params, x)
        
        return output.squeeze(0)
    d1u = grad(f)(X_ij, func_params)   # jacobian of u
    d2u = jacrev(grad(f))(X_ij, func_params) # hessian of u 
    dudX2 = torch.diagonal(d2u)
    
    lhs = -(beta*torch.sum(dudX2)+torch.sum(d1u*gradbeta))  # left hand side
    loss_f_inhom_vc = lhs - Rf_inner
    return loss_f_inhom_vc.flatten()

def compute_loss_inj_vc(func_params, X_interf, U_inj, Normal_ij,Ibeta1,Ibeta2):
    # derivative normal jump on the interface for training U (rawnet), beta(x) is the varible coefficient
    def f(x,z,func_params):
        x=torch.hstack([x,z])
        output = functional_call(compute_loss_inj_vc._model, func_params, x)
        return output.squeeze(0)
    phi=compute_loss_inj_vc._phi
    z=torch.tensor(0.0,dtype=torch.float64,device=X_interf.device)
    df=grad(lambda x:f(x,phi(x),func_params))(X_interf)
    loss_inj =  (Ibeta2-Ibeta1)*torch.sum(df*Normal_ij)+ \
          2 * Ibeta1* grad(f,argnums=1)(X_interf,z,func_params)* torch.norm(grad(phi)(X_interf))- U_inj
    return loss_inj.flatten()
# decoupling approach training V
def compute_loss_fj(func_params, X_inner, Rf_inner):
    ### compute laplace(V)-[f] on the interface, for training patchnet 
    def f(x, func_params):
        output = functional_call(compute_loss_fj._model,func_params, x)

        return output.squeeze(0)
    
    grad2_f = (jacrev(grad(f)))(X_inner, func_params)
    dudX2 = (torch.diagonal(grad2_f))
    
    # laplace = (dudX2[0] + dudX2[1])
    laplace = torch.sum(dudX2)
    
    loss_Res = laplace - Rf_inner

    return loss_Res.flatten()

def compute_loss_normal_jump(func_params, X_ij, Normal_ij, Unj_ij):
    # derivative normal jump for training patchnet 
    def f(x, func_params):
        output = functional_call(compute_loss_normal_jump._model,func_params, x)
        return output.squeeze(0)
    
    grad_f = (grad(f))(X_ij, func_params)
    df = (grad_f)
    normal_jump_pred = torch.sum(Normal_ij*df)
    loss_normal_jump = normal_jump_pred - Unj_ij
        
    return loss_normal_jump.flatten()



#===============================================
# loss for co-training 

def compute_loss_bd_composed(func_params, X_bd, U_bd):
    def f(x, func_params):
        output = functional_call(compute_loss_bd_composed._model,func_params, x)
        return output[0], output[1]
    u_pred , _ = f(X_bd, func_params)
    loss_bd= u_pred - U_bd
    return loss_bd.flatten()
    
def compute_loss_jump_composed(func_params, X_interf, U_w):
    def f(x, func_params):
        output = functional_call(compute_loss_jump_composed._model,func_params, x)
        return output[0], output[1]
    _ , V_pred = f(X_interf, func_params)
    loss_jump = - V_pred - U_w
    return loss_jump.flatten()

def compute_loss_normal_jump_composed(func_params, X_ij, Normal_ij, Unj_ij, iBeta1, iBeta2):
    """
    ((ibeta2-ibeta1)*div(U)- ibeta1*div(V))*normal-u_nj
    """
    def f(x, func_params):
        output = functional_call(compute_loss_normal_jump_composed._model,func_params, x)
        return output[0], output[1]
    grad_U , grad_V = jacrev(f)(X_ij, func_params)
    loss_normal_jump = ((iBeta2-iBeta1)*grad_U-iBeta1*grad_V).dot(Normal_ij) - Unj_ij
    return loss_normal_jump.flatten()

def compute_loss_f_composed(func_params, X_in, U_f, Sign, Beta, Gradbeta):
    def f(x, func_params):
        output = functional_call(compute_loss_f_composed._model,func_params, x)
        return (output[0]+ Sign * output[1]).squeeze(0)
    d1u = grad(f)(X_in, func_params)   # jacobian of u
    d2u = jacrev(grad(f))(X_in, func_params) # hessian of u 
    dudX2 = torch.diagonal(d2u)
    
    lhs = -(Beta*torch.sum(dudX2)+torch.sum(d1u*Gradbeta))  # left hand side
    loss_f_inhom_vc = lhs - U_f
    return loss_f_inhom_vc.flatten()


loss_list=[getattr(sys.modules[__name__],name) for name in dir() if name.startswith("compute_loss")]

def setup_fm(model):
    for loss in loss_list:
        loss._model = model

# use torch.autograd
def jacobian(y,xs,i=0,j=None):
    y = y[:, i : i + 1] if y.shape[1] > 1 else y
    g = torch.autograd.grad(
                  y, xs, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
              )[0]
    return (
            g if j is None or xs.shape[1] == 1 else g[:, j : j + 1]
        )
def hessian(ys, xs, component=None, i=0, j=0, grad_y=None):
    dim_y = ys.shape[1]
    if dim_y > 1:
        if component is None:
            raise ValueError("The component of y is missing.")
        if component >= dim_y:
            raise ValueError(
                "The component of y={} cannot be larger than the dimension={}.".format(
                    component, dim_y
                )
            )
    else:
        if component is not None:
            raise ValueError("Do not use component for 1D y.")
        component = 0

    if grad_y is None:
        grad_y = jacobian(ys, xs, i=component, j=None)
    return jacobian(grad_y, xs, i=i, j=j)

loss_fn = torch.nn.MSELoss(reduction="mean")
def loss_inner(model, x, U_f):
  y=model(x)
  grad_y= jacobian(y,x)
  negative_laplace =0
  for k in range(x.shape[1]):
    negative_laplace=negative_laplace-hessian(y,x,i=k,j=k,grad_y=grad_y)
  return loss_fn(negative_laplace, U_f)
        
def loss_boundary(model, x, U_bd):
  return loss_fn(model(x),U_bd)*50



# test
if __name__=="__main__":
    from model import Shallow_ext
    device = torch.device('cpu')
    torch.set_default_dtype(torch.float64)
    r0=0.4;r1=0.1; w=5
    r=lambda theta:r0+r1*torch.sin(w*theta)

    def phi1(x):
        theta= torch.atan2(x[1],x[0])
        return x[0]**2+x[1]**2 -r(theta)**2
    def af1(x):
        return phi1(x).abs()
    compute_loss_inj._phi=phi1
    model_r = Shallow_ext(2, 40, 2,addiction_features=[af1])
    func_params_r=dict(model_r.named_parameters())
    setup_fm(model_r)
    # func_model_r, func_params_r = make_functional(model_r)
    # setup_fm(func_model_r)
    x=torch.tensor([[2,1],[1,2]],dtype=torch.float64)
    U_inj=torch.tensor([[2],[3]],dtype=torch.float64)
    sign=torch.tensor([[1],[0]],dtype=torch.float64)
    # print(vmap(compute_loss_inj,(None,0,0))(func_params_r,x,U_inj))
    # print(vmap(jacrev(compute_loss_inj),(None,0,0))(func_params_r,x,U_inj))

    beta1= lambda x : 1.0*x[:,0]
    beta2= lambda x : 2.0*x[:,1]
    beta=torch.tensor([[1],[-1]],dtype=torch.float64)
    gradbeta=torch.tensor([[1,1],[-1,1]],dtype=torch.float64)
    print(vmap(compute_loss_f_composed,(None,0,0,0,0,0))(func_params_r,x,U_inj,sign,beta,gradbeta))
    print(vmap(compute_loss_bd_composed,(None,0,0))(func_params_r,x,U_inj))
    print(vmap(compute_loss_jump_composed,(None,0,0))(func_params_r,x,U_inj))

    # compute_loss_f_inhom_vc(func_params_r,x[0],U_inj[0],beta[0],gradbeta[0])
